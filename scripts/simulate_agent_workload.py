"""Simulate a multi-turn agent workload against SGLang's real radix cache
(CPU-only, via ``RadixCache.create_simulated``) and dump a snapshot.

This is the v0.2 end-to-end demo: real SGLang radix code → KVScope snapshot
→ `kvscope analyze`.

Workload shape (mirrors TraceLab's findings about coding agents):
- A long shared system prompt (the "tool schema + instructions" prefix).
- Multiple conversations that share that prefix, then diverge.
- Multi-turn: each turn re-sends the growing conversation (prefix reuse).
- Some short-duration divergences (small nodes = split residue).
- Locking a subtree (in-flight request) to create protected tokens.
"""
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

# Make kvscope importable from the repo checkout.
sys.path.insert(0, "/mnt/d/Project/labs/kvscope")

from kvscope.dump import write_snapshot  # noqa: E402

SYSTEM_PROMPT = list(range(1000, 1000 + 512))  # 512-token shared system prompt
CONV_A = list(range(2000, 2000 + 128))        # conversation A divergence
CONV_B = list(range(3000, 3000 + 96))         # conversation B divergence
TURNS = 4


def _key(tokens: list[int]) -> RadixKey:
    return RadixKey(token_ids=array("q", tokens))


def main(out_path: str) -> None:
    # Mock allocator so evict() can call free_segment without real GPU pools.
    mock_allocator = Mock()
    mock_allocator.device = torch.device("cpu")

    cache = RadixCache.create_simulated(
        disable=False,
        page_size=1,
        enable_kv_cache_events=False,
        mock_allocator=mock_allocator,
    )

    # --- Phase 1: seed the shared system prompt (first request inserts it) ---
    cache.insert(InsertParams(key=_key(SYSTEM_PROMPT), value=None, priority=0))

    # --- Phase 2: conversations sharing the prefix, diverging ---
    for conv, div in ((CONV_A, 128), (CONV_B, 96)):
        for turn in range(TURNS):
            context = SYSTEM_PROMPT + conv[: turn * 32]  # growing context
            # Match first (agent reads cache), then insert the new turn.
            cache.match_prefix(MatchPrefixParams(key=_key(context)))
            if turn == 0:
                # First turn inserts the divergence.
                cache.insert(
                    InsertParams(key=_key(context + conv[:32]), value=None, priority=0)
                )
            else:
                # Later turns: already cached, just extend slightly.
                tail = conv[turn * 32 : turn * 32 + 8]
                cache.insert(
                    InsertParams(key=_key(context + tail), value=None, priority=0)
                )

    # --- Phase 3: a divergent single-shot request (creates a short node) ---
    short_req = SYSTEM_PROMPT + [9999, 9998, 9997]
    cache.insert(InsertParams(key=_key(short_req), value=None, priority=0))

    # --- Phase 4: lock the shared subtree (in-flight request) ---
    match = cache.match_prefix(MatchPrefixParams(key=_key(SYSTEM_PROMPT)))
    cache.inc_lock_ref(match.last_device_node)

    # --- Phase 5: evict some tokens to create evicted nodes ---
    cache.evict(EvictParams(num_tokens=20))

    # --- Phase 6: dump ---
    # NOTE: with a Mock allocator there are no real pool stats; report
    # the radix tree's own accounting instead (evictable + protected).
    evictable = cache.evictable_size()
    protected = cache.protected_size()
    pool_stats = {
        "total_tokens": evictable + protected,
        "used_tokens": evictable + protected,
        "evictable_tokens": evictable,
        "protected_tokens": protected,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        n = write_snapshot(cache, f, extra_pool=pool_stats)
    print(f"dumped {n} records -> {out_path}")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/kvscope-demo.jsonl"
    main(out)
