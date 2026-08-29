"""Simulate a multi-turn agent workload against SGLang's real radix cache
(CPU-only, via ``RadixCache.create_simulated``), dumping BOTH:
  1. a radix-tree snapshot (consumed by ``kvscope analyze``)
  2. per-turn KV-reuse records (consumed by ``kvscope turns``)

Workload shape (mirrors TraceLab's findings about coding agents):
- A long shared system prompt (the "tool schema + instructions" prefix).
- Multiple conversations sharing that prefix, then diverging.
- Multi-turn: each turn re-sends the growing conversation (prefix reuse),
  so early turns should hit almost everything and reuse should stay high.
- One cold conversation to show the miss contrast.
"""
import json
import sys
from array import array
from unittest.mock import Mock

import torch

from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.mem_cache.base_prefix_cache import (
    InsertParams,
    MatchPrefixParams,
)

sys.path.insert(0, "/mnt/d/Project/labs/kvscope")

from kvscope.dump import write_snapshot  # noqa: E402
from kvscope.turns import TurnRecord, compute_hit_length_from_node  # noqa: E402

SYSTEM_PROMPT = list(range(1000, 1000 + 512))   # 512-token shared system prompt
CONV_A = list(range(2000, 2000 + 128))          # conversation A divergence
CONV_B = list(range(3000, 3000 + 96))           # conversation B divergence
CONV_COLD = list(range(7000, 7000 + 64))        # cold conversation (no shared prefix)
TURNS = 4


def _key(tokens):
    return RadixKey(token_ids=array("q", tokens))


def _record_turn(turns, cache, turn_no, tokens, conv_name):
    """Match + record hit data for one turn."""
    match = cache.match_prefix(MatchPrefixParams(key=_key(tokens)))
    hit_len = compute_hit_length_from_node(match.last_device_node)
    # Insert the (already-present) context so the tree is complete; the
    # scheduler normally inserts only the new part, but inserting the whole
    # context is idempotent for reuse accounting (it reuses the prefix).
    cache.insert(InsertParams(key=_key(tokens), value=None, priority=0))
    turns.append(
        TurnRecord(
            turn=turn_no,
            context_tokens=len(tokens),
            hit_length=hit_len,
            new_tokens=len(tokens) - hit_len,
            hit_node_id=match.last_device_node.id,
        )
    )
    return hit_len


def main(out_prefix: str) -> None:
    mock_allocator = Mock()
    mock_allocator.device = torch.device("cpu")
    cache = RadixCache.create_simulated(
        disable=False, page_size=1, enable_kv_cache_events=False,
        mock_allocator=mock_allocator,
    )

    turns: list[TurnRecord] = []

    # Warm up with the system prompt (first request inserts it).
    cache.insert(InsertParams(key=_key(SYSTEM_PROMPT), value=None, priority=0))

    # --- Warm conversations (shared prefix, growing context) ---
    for conv_name, conv in (("A", CONV_A), ("B", CONV_B)):
        for turn in range(1, TURNS + 1):
            context = SYSTEM_PROMPT + conv[: turn * 32]
            _record_turn(turns, cache, turn, context, conv_name)

    # --- Cold conversation (no shared prefix -> first turn is a full miss) ---
    for turn in range(1, TURNS + 1):
        context = CONV_COLD[: turn * 16]
        _record_turn(turns, cache, turn, context, "COLD")

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
                    }
                )
                + "\n"
            )
    print(f"turns: {len(turns)} records -> {turns_path}")


if __name__ == "__main__":
    prefix = sys.argv[1] if len(sys.argv) > 1 else "/tmp/kvscope-demo"
    main(prefix)
