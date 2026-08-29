# SPDX-License-Identifier: Apache-2.0
"""Tests for turns.py: per-turn KV-reuse analysis (synthetic records)."""

from __future__ import annotations

import io
import json

import pytest

from kvscope.turns import (
    TurnRecord,
    compute_hit_length_from_node,
    load_turns,
    summarize_turns,
)


class FakeKey:
    def __init__(self, n):
        self._n = n

    def __len__(self):
        return self._n


class FakeNode:
    _counter = 0

    def __init__(self, token_len, parent=None):
        self.id = FakeNode._counter
        FakeNode._counter += 1
        self.key = FakeKey(token_len) if token_len > 0 else None
        self.parent = parent


def test_compute_hit_length_chain():
    FakeNode._counter = 0
    root = FakeNode(0)          # id 0, parent None (true root)
    n1 = FakeNode(100, root)    # id 1
    n2 = FakeNode(30, n1)       # id 2
    # chain: root(0) <- n1(100) <- n2(30); sum = 130
    assert compute_hit_length_from_node(n2) == 130
    assert compute_hit_length_from_node(n1) == 100
    assert compute_hit_length_from_node(root) == 0


def test_summarize_mixed_hits():
    records = [
        TurnRecord(turn=1, context_tokens=100, hit_length=90, new_tokens=10, hit_node_id=1),
        TurnRecord(turn=2, context_tokens=100, hit_length=50, new_tokens=50, hit_node_id=1),
        TurnRecord(turn=3, context_tokens=100, hit_length=0, new_tokens=100, hit_node_id=0),
    ]
    s = summarize_turns(records)
    assert s.num_turns == 3
    assert s.total_context_tokens == 300
    assert s.total_hit_tokens == 140
    assert s.total_new_tokens == 160
    assert s.avg_hit_ratio == pytest.approx(140 / 300)
    assert s.final_hit_ratio == 0.0
    assert s.worst_turn == 3
    assert s.best_turn == 1
    assert s.hit_ratio_trend == [0.9, 0.5, 0.0]


def test_summarize_empty():
    s = summarize_turns([])
    assert s.num_turns == 0
    assert s.avg_hit_ratio == 0.0


def test_turn_record_properties():
    r = TurnRecord(turn=1, context_tokens=200, hit_length=150, new_tokens=50, hit_node_id=1)
    assert r.hit_ratio == 0.75
    assert r.recompute_cost_ratio == 0.25


def test_load_turns_jsonl(tmp_path):
    p = tmp_path / "turns.jsonl"
    p.write_text(
        "\n".join(
            json.dumps(
                {"turn": i, "context_tokens": 100, "hit_length": 80,
                 "new_tokens": 20, "hit_node_id": 1}
            )
            for i in (1, 2)
        )
        + "\n",
        encoding="utf-8",
    )
    records = load_turns(p)
    assert len(records) == 2
    assert records[0].turn == 1
    assert records[0].hit_length == 80


def test_load_turns_missing_field(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"turn": 1, "context_tokens": 100}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="missing required field"):
        load_turns(p)
