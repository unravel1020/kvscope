# SPDX-License-Identifier: Apache-2.0
"""Per-turn KV-reuse analysis for multi-turn (agent) workloads.

A turn is one LLM call in a conversation: the client sends the full
growing context (system prompt + history + new input), the server matches
the longest cached prefix, computes the new tokens, and inserts them.

KVScope turns analysis answers: for each turn, how much of the context was
reused from cache (hit length), how much had to be recomputed (new tokens),
and how does reuse evolve over the conversation (the "context growth vs
cache reuse" curve that dominates agent serving cost — see TraceLab:
coding-agent workloads have median 119K-token prefixes with tiny appends).

The ``hit_length`` for a turn is computed by walking the matched node's
parent chain and summing ``token_len`` — this mirrors SGLang's radix-tree
structure exactly (a node's tokens are the shared prefix of all its
descendants).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class TurnRecord:
    """One LLM call in a conversation."""

    turn: int  # 1-based
    context_tokens: int  # total tokens sent in this turn (prompt + history + new)
    hit_length: int  # tokens reused from cache (matched prefix length)
    new_tokens: int  # tokens that had to be computed (context_tokens - hit_length)
    hit_node_id: int  # matched node id (0 = root = cold miss)
    reused_after_split: bool = False  # True when the match ended mid-node (split)

    @property
    def hit_ratio(self) -> float:
        if self.context_tokens == 0:
            return 0.0
        return self.hit_length / self.context_tokens

    @property
    def recompute_cost_ratio(self) -> float:
        """Fraction of this turn's context that was recomputed (wasted)."""
        return 1.0 - self.hit_ratio


@dataclass
class TurnSummary:
    """Aggregate metrics over a multi-turn session."""

    num_turns: int
    total_context_tokens: int
    total_hit_tokens: int
    total_new_tokens: int
    avg_hit_ratio: float
    final_hit_ratio: float
    hit_ratio_trend: list[float]  # per-turn hit ratio
    recompute_total: int  # total tokens recomputed across turns
    worst_turn: int  # turn index with lowest hit ratio
    best_turn: int

    def to_dict(self) -> dict:
        return {
            "num_turns": self.num_turns,
            "total_context_tokens": self.total_context_tokens,
            "total_hit_tokens": self.total_hit_tokens,
            "total_new_tokens": self.total_new_tokens,
            "avg_hit_ratio": round(self.avg_hit_ratio, 4),
            "final_hit_ratio": round(self.final_hit_ratio, 4),
            "hit_ratio_trend": [round(r, 4) for r in self.hit_ratio_trend],
            "recompute_total": self.recompute_total,
            "worst_turn": self.worst_turn,
            "best_turn": self.best_turn,
        }

    def render_text(self) -> str:
        lines = [
            "Per-Turn KV Reuse Report",
            "=" * 40,
            f"turns                 : {self.num_turns}",
            f"total context tokens  : {self.total_context_tokens}",
            f"total hit tokens      : {self.total_hit_tokens}",
            f"total new tokens      : {self.total_new_tokens}",
            f"avg hit ratio         : {self.avg_hit_ratio:.2%}",
            f"final hit ratio       : {self.final_hit_ratio:.2%}",
            f"recomputed total      : {self.recompute_total}",
            f"worst turn            : {self.worst_turn}",
            f"best turn             : {self.best_turn}",
            "",
            "per-turn hit ratio:",
        ]
        for i, r in enumerate(self.hit_ratio_trend, start=1):
            bar = "#" * int(r * 30)
            lines.append(f"  turn {i:>3}: {r:6.1%} |{bar}")
        return "\n".join(lines)


def compute_hit_length_from_node(node: Any) -> int:
    """Sum ``token_len`` along the parent chain up to (not including) root.

    Mirrors how SGLang's match result's ``last_device_node`` represents the
    matched prefix: the path from root to that node IS the shared prefix.
    """
    total = 0
    cur = node
    seen: set[int] = set()
    while cur is not None and getattr(cur, "id", None) not in seen:
        seen.add(getattr(cur, "id", id(cur)))
        parent = getattr(cur, "parent", None)
        if parent is not None:
            key = getattr(cur, "key", None)
            if key is not None:
                try:
                    total += len(key)
                except TypeError:
                    total += len(key)
        cur = parent
    return total


def _is_split_hit(node: Any) -> bool:
    """A match ends mid-node (split) when the node has remaining tokens
    beyond the match — approximated here by checking if the matched node's
    parent chain hit length is used. We can't know from the node alone;
    callers pass ``split`` explicitly when they observe it."""
    return False


def load_turns(path: str | Path) -> list[TurnRecord]:
    """Parse a turns JSONL file.

    Each line: {"turn": int, "context_tokens": int, "hit_length": int,
                "new_tokens": int, "hit_node_id": int}
    """
    records: list[TurnRecord] = []
    with open(path, encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            required = {"turn", "context_tokens", "hit_length"}
            missing = required - set(obj)
            if missing:
                raise ValueError(
                    f"{path}:{line_no + 1}: missing required field(s): {sorted(missing)}"
                )
            records.append(
                TurnRecord(
                    turn=int(obj["turn"]),
                    context_tokens=int(obj["context_tokens"]),
                    hit_length=int(obj["hit_length"]),
                    new_tokens=int(obj.get("new_tokens", 0)),
                    hit_node_id=int(obj.get("hit_node_id", 0)),
                )
            )
    return records


def summarize_turns(records: list[TurnRecord]) -> TurnSummary:
    """Aggregate per-turn records into a session summary."""
    if not records:
        return TurnSummary(
            num_turns=0,
            total_context_tokens=0,
            total_hit_tokens=0,
            total_new_tokens=0,
            avg_hit_ratio=0.0,
            final_hit_ratio=0.0,
            hit_ratio_trend=[],
            recompute_total=0,
            worst_turn=0,
            best_turn=0,
        )

    total_ctx = sum(r.context_tokens for r in records)
    total_hit = sum(r.hit_length for r in records)
    total_new = sum(r.new_tokens for r in records)
    trend = [r.hit_ratio for r in records]

    ratios = [r.hit_ratio for r in records]
    worst = min(range(len(records)), key=lambda i: ratios[i]) + 1
    best = max(range(len(records)), key=lambda i: ratios[i]) + 1

    return TurnSummary(
        num_turns=len(records),
        total_context_tokens=total_ctx,
        total_hit_tokens=total_hit,
        total_new_tokens=total_new,
        avg_hit_ratio=(total_hit / total_ctx) if total_ctx else 0.0,
        final_hit_ratio=trend[-1] if trend else 0.0,
        hit_ratio_trend=trend,
        recompute_total=total_new,
        worst_turn=worst,
        best_turn=best,
    )
