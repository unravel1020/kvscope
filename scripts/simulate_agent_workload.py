"""Simulate a multi-turn agent workload against SGLang's real radix cache
(CPU-only, via ``RadixCache.create_simulated``), dumping THREE artifacts:
  1. a radix-tree snapshot (consumed by ``kvscope analyze``)
  2. per-turn KV-reuse records (consumed by ``kvscope turns``)
  3. a KV placement event stream (consumed by ``kvscope events``)

The event stream mirrors SGLang's ``BlockStored``/``BlockRemoved``/
``AllBlocksCleared`` wire format (block_hashes / parent_block_hash /
medium), but uses a deterministic pure-Python hash instead of the
``HiCache native hash`` C++ extension, so this runs on any CPU without
building native modules. The radix tree itself is SGLang's real code.

Workload shape (mirrors TraceLab's findings about coding agents):
- A long shared system prompt (the "tool schema + instructions" prefix).
- Multiple conversations sharing that prefix, then diverging.
- Multi-turn: each turn re-sends the growing conversation (prefix reuse),
  so early turns should hit almost everything and reuse should stay high.
- One cold conversation to show the miss contrast.
- A small eviction to produce BlockRemoved events.
"""
import hashlib
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

sys.path.insert(0, "/mnt/d/Project/labs/kvscope")

from kvscope.dump import write_snapshot  # noqa: E402
from kvscope.turns import TurnRecord, compute_hit_length_from_node  # noqa: E402

# Gap-duration simulation knobs (TraceLab-informed):
# - most inter-turn gaps are short (seconds), matching human reading/typing
# - occasional long gaps (>= 5 min) where the cache starts to decay
SHORT_GAP_SECONDS = (5, 60)       # uniform range for short gaps
LONG_GAP_SECONDS = (300, 1800)    # 5 min .. 30 min
LONG_GAP_PROBABILITY = 0.15       # ~15% of gaps are long
CACHE_DECAY_SECONDS = 300         # gaps >= this evict stale prefixes
DECAY_EVICT_TOKENS = 512          # tokens evicted after a long gap


def _block_hash(tokens: list[int]) -> int:
    """Deterministic pure-Python block hash (stand-in for SGLang's native
    page hash). Stable across runs; NOT cryptographically meaningful."""
    payload = ",".join(str(t) for t in tokens)
    return int(hashlib.sha256(payload.encode()).hexdigest()[:16], 16)

SYSTEM_PROMPT = list(range(1000, 1000 + 512))   # 512-token shared system prompt
CONV_A = list(range(2000, 2000 + 128))          # conversation A divergence
CONV_B = list(range(3000, 3000 + 96))           # conversation B divergence
CONV_COLD = list(range(7000, 7000 + 64))        # cold conversation (no shared prefix)
TURNS = 5
N_CONVERSATIONS = 6                             # more convs -> more gap samples


def _key(tokens):
    return RadixKey(token_ids=array("q", tokens))


def _record_turn(turns, cache, turn_no, tokens, collector, conv_name, ts, gap_s):
    """Match + record hit data for one turn (with timestamp + gap)."""
    match = cache.match_prefix(MatchPrefixParams(key=_key(tokens)))
    hit_len = compute_hit_length_from_node(match.last_device_node)
    result = cache.insert(InsertParams(key=_key(tokens), value=None, priority=0))
    if result.last_device_node is not None:
        collector.store(result.last_device_node)
    turns.append(
        TurnRecord(
            turn=turn_no,
            context_tokens=len(tokens),
            hit_length=hit_len,
            new_tokens=len(tokens) - hit_len,
            hit_node_id=match.last_device_node.id,
        )
    )
    # Attach timing to the raw record we emit later (turns JSONL).
    turns[-1].timestamp = ts
    turns[-1].gap_seconds = gap_s
    return hit_len


def _next_gap(rng: random.Random) -> float:
    """Sample an inter-turn gap (seconds)."""
    if rng.random() < LONG_GAP_PROBABILITY:
        return rng.uniform(*LONG_GAP_SECONDS)
    return rng.uniform(*SHORT_GAP_SECONDS)


class EventCollector:
    """Collects store/remove events in SGLang's wire format, using the
    deterministic pure-Python block hash. Mirrors KVCacheEventRecorder's
    event shape so kvscope's events module can consume them."""

    def __init__(self, page_size: int = 1):
        self.page_size = page_size
        self.events: list[dict] = []

    def store(self, node):
        """Record a BlockStored-style event for a tree node."""
        n_tok = len(node.key) if node.key is not None else 0
        if n_tok == 0:
            return
        raw = list(node.key.token_ids) if hasattr(node.key, "token_ids") else []
        for start in range(0, n_tok, self.page_size):
            end = min(start + self.page_size, n_tok)
            page_tokens = raw[start:end]
            if not page_tokens:
                continue
            h = _block_hash(page_tokens)
            parent = node.parent
            parent_hash = None
            if parent is not None and parent.key is not None and len(parent.key) > 0:
                parent_raw = list(parent.key.token_ids) if hasattr(parent.key, "token_ids") else []
                parent_hash = _block_hash(parent_raw[-self.page_size:])
            self.events.append(
                {
                    "type": "stored",
                    "block_hashes": [h],
                    "parent_block_hash": parent_hash,
                    "num_tokens": len(page_tokens),
                    "medium": "GPU",
                }
            )

    def remove(self, node):
        """Record a BlockRemoved-style event for a tree node."""
        n_tok = len(node.key) if node.key is not None else 0
        if n_tok == 0:
            return
        raw = list(node.key.token_ids) if hasattr(node.key, "token_ids") else []
        hashes = []
        for start in range(0, n_tok, self.page_size):
            end = min(start + self.page_size, n_tok)
            page_tokens = raw[start:end]
            if page_tokens:
                hashes.append(_block_hash(page_tokens))
        if hashes:
            self.events.append(
                {"type": "removed", "block_hashes": hashes, "medium": "GPU"}
            )


def main(out_prefix: str) -> None:
    mock_allocator = Mock()
    mock_allocator.device = torch.device("cpu")
    cache = RadixCache.create_simulated(
        disable=False, page_size=1, enable_kv_cache_events=False,
        mock_allocator=mock_allocator,
    )
    collector = EventCollector(page_size=1)

    turns: list[TurnRecord] = []
    rng = random.Random(42)  # deterministic
    clock = 0.0  # simulated wall clock (seconds)

    # Warm up with the system prompt (first request inserts it).
    r = cache.insert(InsertParams(key=_key(SYSTEM_PROMPT), value=None, priority=0))
    if r.last_device_node is not None:
        collector.store(r.last_device_node)

    # --- Warm conversations (shared prefix, growing context) ---
    convs = [(f"A{i}", list(range(2000 + i * 100, 2000 + i * 100 + 128))) for i in range(N_CONVERSATIONS)]
    for conv_name, conv in convs:
        for turn in range(1, TURNS + 1):
            gap = _next_gap(rng)
            clock += gap
            # Long gap: simulate cache decay — evict ALL evictable leaves
            # (in reality LRU would have evicted idle prefixes during the
            # human's absence; the shared root stays locked so the system
            # prompt survives, but the conversation divergences go cold).
            if gap >= CACHE_DECAY_SECONDS:
                stale = list(getattr(cache, "evictable_leaves", []))
                for node in stale:
                    collector.remove(node)
                cache.evict(EvictParams(num_tokens=10**9))
            context = SYSTEM_PROMPT + conv[: turn * 32]
            _record_turn(turns, cache, turn, context, collector, conv_name, clock, gap)

    # --- Cold conversation (no shared prefix -> first turn is a full miss) ---
    for turn in range(1, TURNS + 1):
        gap = _next_gap(rng)
        clock += gap
        if gap >= CACHE_DECAY_SECONDS:
            stale = list(getattr(cache, "evictable_leaves", []))
            for node in stale:
                collector.remove(node)
            cache.evict(EvictParams(num_tokens=10**9))
        context = CONV_COLD[: turn * 16]
        _record_turn(turns, cache, turn, context, collector, "COLD", clock, gap)

    # --- Evict some tokens; record remove events for evicted leaves ---
    evictable_leaves = list(getattr(cache, "evictable_leaves", []))
    for node in evictable_leaves:
        collector.remove(node)
    cache.evict(EvictParams(num_tokens=40))

    # --- Dump snapshot ---
    evictable = cache.evictable_size()
    protected = cache.protected_size()
    pool_stats = {
        "total_tokens": evictable + protected,
        "used_tokens": evictable + protected,
        "evictable_tokens": evictable,
        "protected_tokens": protected,
    }
    snap_path = out_prefix + ".snapshot.jsonl"
    with open(snap_path, "w", encoding="utf-8") as f:
        n = write_snapshot(cache, f, extra_pool=pool_stats)
    print(f"snapshot: {n} records -> {snap_path}")

    # --- Dump turns ---
    turns_path = out_prefix + ".turns.jsonl"
    with open(turns_path, "w", encoding="utf-8") as f:
        for t in turns:
            f.write(
                json.dumps(
                    {
                        "turn": t.turn,
                        "context_tokens": t.context_tokens,
                        "hit_length": t.hit_length,
                        "new_tokens": t.new_tokens,
                        "hit_node_id": t.hit_node_id,
                        "timestamp": getattr(t, "timestamp", 0.0),
                        "gap_seconds": getattr(t, "gap_seconds", 0.0),
                    }
                )
                + "\n"
            )
    print(f"turns: {len(turns)} records -> {turns_path}")

    # --- Dump KV event stream ---
    events_path = out_prefix + ".events.jsonl"
    with open(events_path, "w", encoding="utf-8") as f:
        for ev in collector.events:
            f.write(json.dumps(ev) + "\n")
    print(f"events: {len(collector.events)} records -> {events_path}")


if __name__ == "__main__":
    prefix = sys.argv[1] if len(sys.argv) > 1 else "/tmp/kvscope-demo"
    main(prefix)
