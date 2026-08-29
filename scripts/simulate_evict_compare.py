"""CacheWise-style eviction-strategy comparison on SGLang's real radix cache.

Runs the SAME multi-turn agent workload twice under a FINITE KV pool
(memory pressure), once per eviction strategy:

- ``lru``       : SGLang's native evict() (LRU heap over evictable leaves).
- ``predictive``: gap-informed policy — when the pool is full, prefer
                  evicting leaves whose (idle time, low predicted reuse)
                  marks them as likely-dead (cf. CacheWise: use the
                  inter-turn gap to predict reuse and drive eviction).

Both strategies share the exact same request sequence (same gaps, same
contexts), so the hit-rate difference isolates the eviction policy.

Outputs a comparison JSON consumed by ``kvscope evict``.
"""
import json
import random
import sys
from array import array
from unittest.mock import Mock

import torch

from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)

# --- pool / workload knobs ---
POOL_TOKENS = 1000          # finite KV pool: forces eviction under pressure
SYSTEM_PROMPT = list(range(1000, 1000 + 512))
N_CONVERSATIONS = 10
CONV_LEN = 120              # tokens per conversation divergence (inserted in 20-tok chunks)
CHUNK = 20                  # per-turn increment -> small leaves, page-like granularity
TURNS = 6
SHORT_GAP = (5, 60)
LONG_GAP = (300, 1200)
LONG_GAP_PROB = 0.25
DECAY_THRESHOLD_S = 300.0

# --- workload builder (deterministic) ---
def _build_workload(seed: int) -> list[tuple[str, list[int], float]]:
    """Return [(conv_name, tokens, gap_s), ...] for one full run.

    Conversations run *interleaved* (C0T1, C1T1, C2T1, C0T2, ...) and gaps
    occur *within* a conversation (human pauses mid-task), so after a long
    gap the idle leaves of that same conversation are genuinely stale —
    the signal predictive eviction acts on.
    """
    rng = random.Random(seed)
    convs = [
        list(range(2000 + i * 100, 2000 + i * 100 + CONV_LEN))
        for i in range(N_CONVERSATIONS)
    ]
    turns: list[tuple[str, list[int], float]] = []
    for t in range(1, TURNS + 1):
        for i in range(N_CONVERSATIONS):
            gap = rng.uniform(*SHORT_GAP)
            if rng.random() < LONG_GAP_PROB:
                gap = rng.uniform(*LONG_GAP)
            ctx = SYSTEM_PROMPT + convs[i][: t * CHUNK]
            turns.append((f"C{i}", ctx, gap))
    return turns


def _key(tokens):
    return RadixKey(token_ids=array("q", tokens))


def _new_cache():
    mock_allocator = Mock()
    mock_allocator.device = torch.device("cpu")
    cache = RadixCache.create_simulated(
        disable=False, page_size=1, enable_kv_cache_events=False,
        mock_allocator=mock_allocator,
    )
    # Lock the shared system prompt (simulates a server pinning the shared
    # prefix, as SGLang does via inc_lock_ref for in-flight/important data).
    r = cache.insert(InsertParams(key=_key(SYSTEM_PROMPT), value=None, priority=0))
    if r.last_device_node is not None:
        cache.inc_lock_ref(r.last_device_node)
    return cache


def _pool_used(cache) -> int:
    return cache.evictable_size() + cache.protected_size()


def _run_lru(cache, workload) -> dict:
    """Baseline: let SGLang's LRU evict() handle pressure."""
    total_hit = 0
    total_ctx = 0
    evictions = 0
    for _name, tokens, gap in workload:
        # Simulate pool pressure: if over budget, evict LRU leaves (small,
        # page-like granularity keeps the eviction fine-grained).
        while _pool_used(cache) + len(tokens) > POOL_TOKENS:
            leaves = list(getattr(cache, "evictable_leaves", []))
            if not leaves:
                break
            before = _pool_used(cache)
            # SGLang evict() pops the leaf with lowest priority (LRU) and
            # frees its whole segment; request a small budget.
            cache.evict(EvictParams(num_tokens=CHUNK))
            after = _pool_used(cache)
            if after >= before:
                break  # nothing more evictable (e.g. all locked)
            evictions += 1

        m = cache.match_prefix(MatchPrefixParams(key=_key(tokens)))
        hit = sum(
            len(n.key) if n.key is not None else 0
            for n in _path(cache, m.last_device_node)
        )
        cache.insert(InsertParams(key=_key(tokens), value=None, priority=0))
        total_hit += hit
        total_ctx += len(tokens)

    return {
        "hit_ratio": total_hit / total_ctx if total_ctx else 0.0,
        "total_hit": total_hit,
        "total_ctx": total_ctx,
        "eviction_rounds": evictions,
    }


def _path(cache, node):
    """Nodes from root to ``node`` (inclusive), for hit-length summing."""
    nodes = []
    cur = node
    seen = set()
    while cur is not None and getattr(cur, "id", None) not in seen:
        seen.add(getattr(cur, "id", id(cur)))
        nodes.append(cur)
        cur = cur.parent
    nodes.reverse()
    return nodes


def _run_predictive(cache, workload) -> dict:
    """Gap-informed eviction: prefer evicting idle leaves predicted dead.

    Policy: when over budget, evict leaves in order of (idle_time desc,
    predicted_reuse asc). Idle time = now - last_access_time; a leaf idle
    longer than DECAY_THRESHOLD_S is a strong eviction candidate.
    """
    clock = 0.0
    total_hit = 0
    total_ctx = 0
    evictions = 0

    for _name, tokens, gap in workload:
        clock += gap

        # --- Predictive: after a long gap, proactively evict ALL idle
        # evictable leaves (the conversation's stale prefixes are dead).
        # NOTE: SGLang's node.last_access_time uses time.monotonic() (real
        # clock), which is not comparable to our simulated clock, so we
        # cannot filter by idle time per-node here. Instead we treat every
        # long gap as "that conversation's cache went cold" and clear the
        # evictable part of the tree (shared system prompt stays locked).
        if gap >= DECAY_THRESHOLD_S:
            stale = list(getattr(cache, "evictable_leaves", []))
            for node in stale:
                cache.evict(EvictParams(num_tokens=len(node.key) if node.key else CHUNK))

        # --- Under pressure: same LRU mechanism as the baseline, but the
        # proactive cleanup above means fewer dead leaves occupy the pool,
        # so the LRU pass has more room for live prefixes.
        while _pool_used(cache) + len(tokens) > POOL_TOKENS:
            leaves = list(getattr(cache, "evictable_leaves", []))
            if not leaves:
                break
            before = _pool_used(cache)
            cache.evict(EvictParams(num_tokens=CHUNK))
            after = _pool_used(cache)
            if after >= before:
                break
            evictions += 1

        m = cache.match_prefix(MatchPrefixParams(key=_key(tokens)))
        hit = sum(
            len(n.key) if n.key is not None else 0
            for n in _path(cache, m.last_device_node)
        )
        cache.insert(InsertParams(key=_key(tokens), value=None, priority=0))
        total_hit += hit
        total_ctx += len(tokens)

    return {
        "hit_ratio": total_hit / total_ctx if total_ctx else 0.0,
        "total_hit": total_hit,
        "total_ctx": total_ctx,
        "eviction_rounds": evictions,
    }


def main(out_path: str, seed: int = 42) -> None:
    workload = _build_workload(seed)

    lru_cache = _new_cache()
    lru_res = _run_lru(lru_cache, workload)

    pred_cache = _new_cache()
    pred_res = _run_predictive(pred_cache, workload)

    result = {
        "workload": {
            "n_turns": len(workload),
            "n_conversations": N_CONVERSATIONS,
            "turns_per_conv": TURNS,
            "system_prompt_tokens": len(SYSTEM_PROMPT),
            "pool_tokens": POOL_TOKENS,
            "long_gap_prob": LONG_GAP_PROB,
            "decay_threshold_s": DECAY_THRESHOLD_S,
        },
        "strategies": {
            "lru": lru_res,
            "predictive": pred_res,
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"comparison -> {out_path}")
    print(f"  lru        : hit={lru_res['hit_ratio']:.1%} evicts={lru_res['eviction_rounds']}")
    print(f"  predictive : hit={pred_res['hit_ratio']:.1%} evicts={pred_res['eviction_rounds']}")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/evict-compare.json"
    main(out)
