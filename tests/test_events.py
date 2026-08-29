# SPDX-License-Identifier: Apache-2.0
"""Tests for events.py: KV event-stream parsing and analysis."""

from __future__ import annotations

import json

import pytest

from kvscope.events import EventStream, load_events


def _write(tmp_path, records):
    p = tmp_path / "events.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return p


def test_load_basic_stream(tmp_path):
    p = _write(tmp_path, [
        {"type": "stored", "block_hashes": [1], "parent_block_hash": None, "num_tokens": 8, "medium": "GPU"},
        {"type": "stored", "block_hashes": [2], "parent_block_hash": 1, "num_tokens": 8, "medium": "GPU"},
        {"type": "stored", "block_hashes": [1], "parent_block_hash": None, "num_tokens": 8, "medium": "GPU"},
        {"type": "removed", "block_hashes": [1], "medium": "GPU"},
    ])
    s = load_events(p)
    assert len(s.events) == 4
    assert s.num_stored == 3
    assert s.num_removed == 1
    assert s.total_stored_tokens == 24
    # churn: block 1 stored then removed -> 1 of 2 unique stored blocks
    assert s.blocks_stored == {1, 2}
    assert s.blocks_removed == {1}
    assert s.churn_rate == 0.5
    # reuse: block 1 stored twice
    assert s.reuse_confirmed == 1
    assert s.chained_stores == 1  # one store had a parent_block_hash


def test_load_cleared(tmp_path):
    p = _write(tmp_path, [
        {"type": "stored", "block_hashes": [1], "num_tokens": 4, "medium": "GPU"},
        {"type": "cleared"},
    ])
    s = load_events(p)
    assert s.num_cleared == 1
    assert s.num_stored == 1


def test_load_unknown_type(tmp_path):
    p = _write(tmp_path, [{"type": "wat"}])
    with pytest.raises(ValueError, match="unknown event type"):
        load_events(p)


def test_load_empty(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    s = load_events(p)
    assert len(s.events) == 0
    assert s.churn_rate == 0.0


def test_removal_timeline(tmp_path):
    p = _write(tmp_path, [
        {"type": "stored", "block_hashes": [1], "num_tokens": 8, "medium": "GPU"},
        {"type": "removed", "block_hashes": [1], "medium": "GPU"},
    ])
    s = load_events(p)
    assert len(s.removal_timeline) == 1
    seq, h, medium = s.removal_timeline[0]
    assert h == 1
    assert medium == "GPU"
    assert seq == 1  # second record


def test_medium_breakdown(tmp_path):
    p = _write(tmp_path, [
        {"type": "stored", "block_hashes": [1], "num_tokens": 8, "medium": "GPU"},
        {"type": "stored", "block_hashes": [2], "num_tokens": 16, "medium": "CPU_PINNED"},
        {"type": "removed", "block_hashes": [2], "medium": "CPU_PINNED"},
    ])
    s = load_events(p)
    assert s.medium_stored_tokens == {"GPU": 8, "CPU_PINNED": 16}
    assert s.medium_removed_tokens == {"CPU_PINNED": 1}  # 1 block removed
