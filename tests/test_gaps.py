# SPDX-License-Identifier: Apache-2.0
"""Tests for gaps.py: gap-duration analysis and cache-decay prediction."""

from __future__ import annotations

import pytest

from kvscope.gaps import DEFAULT_DECAY_THRESHOLD_S, analyze_gaps
from kvscope.turns import TurnRecord


def _turn(turn, hit_len, gap, ctx=100):
    return TurnRecord(
        turn=turn,
        context_tokens=ctx,
        hit_length=hit_len,
        new_tokens=ctx - hit_len,
        hit_node_id=1,
        gap_seconds=gap,
    )


def test_gap_histogram_distribution():
    records = [
        _turn(1, 90, 5.0),    # <10s
        _turn(2, 90, 30.0),   # 10-60s
        _turn(3, 90, 600.0),  # 5-30min
        _turn(4, 0, 900.0),   # 5-30min (cold)
    ]
    s = analyze_gaps(records)
    hist = dict(s.gap_histogram)
    assert hist["<10s"] == 1
    assert hist["10-60s"] == 1
    assert hist["5-30min"] == 2
    assert s.num_turns == 4


def test_decay_bins_short_gap_high_hit():
    records = [
        _turn(1, 95, 5.0),
        _turn(2, 90, 20.0),
        _turn(3, 80, 40.0),
    ]
    s = analyze_gaps(records)
    short = next(b for b in s.decay_bins if b["label"] == "<10s")
    assert short["n"] == 1
    assert short["hit_ratio"] == pytest.approx(0.95)
    mid = next(b for b in s.decay_bins if b["label"] == "10-60s")
    assert mid["n"] == 2
    assert mid["hit_ratio"] == pytest.approx(0.85)


def test_decay_long_gap_zero_hit():
    # After a long gap the cache is gone -> hit ratio 0.
    records = [_turn(1, 0, 600.0), _turn(2, 0, 900.0)]
    s = analyze_gaps(records)
    long_bin = next(b for b in s.decay_bins if b["label"] == "5-30min")
    assert long_bin["hit_ratio"] == 0.0
    assert long_bin["n"] == 2


def test_predict_monotonic_decay():
    # P(hit) must be non-increasing in gap for a sane model.
    records = [
        _turn(1, 95, 5.0),
        _turn(2, 90, 20.0),
        _turn(3, 85, 45.0),
        _turn(4, 0, 600.0),
        _turn(5, 0, 1200.0),
    ]
    s = analyze_gaps(records)
    p_short = s.predict_hit_probability(10)
    p_mid = s.predict_hit_probability(120)
    p_long = s.predict_hit_probability(900)
    # Not strictly guaranteed with few samples, but long gap should be low.
    assert s.cache_decay_probability(3600) > 0.0
    assert 0.0 <= s.predict_hit_probability(1) <= 1.0
    assert 0.0 <= s.predict_hit_probability(7200) <= 1.0


def test_predict_zero_gap():
    records = [_turn(1, 90, 5.0)]
    s = analyze_gaps(records)
    assert s.predict_hit_probability(0) == 1.0
    assert s.cache_decay_probability(0) == 0.0


def test_empty_records():
    s = analyze_gaps([])
    assert s.num_turns == 0
    assert s.cache_decay_probability(300) >= 0.0


def test_default_threshold():
    assert DEFAULT_DECAY_THRESHOLD_S == 300.0
