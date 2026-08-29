# SPDX-License-Identifier: Apache-2.0
"""Tests for evict.py: eviction-strategy comparison report."""

from __future__ import annotations

import pytest

from kvscope.evict import build_evict_report, load_evict_compare


SAMPLE = {
    "workload": {
        "n_turns": 60,
        "n_conversations": 10,
        "turns_per_conv": 6,
        "system_prompt_tokens": 512,
        "pool_tokens": 1000,
        "long_gap_prob": 0.25,
        "decay_threshold_s": 300.0,
    },
    "strategies": {
        "lru": {"hit_ratio": 0.88, "total_hit": 5280, "total_ctx": 6000,
                "eviction_rounds": 59},
        "predictive": {"hit_ratio": 0.88, "total_hit": 5280, "total_ctx": 6000,
                       "eviction_rounds": 42},
    },
}


def test_build_report_delta():
    r = build_evict_report(SAMPLE)
    assert r["strategies"]["lru"]["eviction_rounds"] == 59
    assert r["strategies"]["predictive"]["eviction_rounds"] == 42
    assert r["delta"]["eviction_rounds"] == -17
    assert r["delta"]["eviction_rounds_pct"] == round(-17 / 59, 4)
    assert r["delta"]["hit_ratio"] == 0.0


def test_build_report_predictive_wins_hit():
    data = {
        **SAMPLE,
        "strategies": {
            "lru": {"hit_ratio": 0.80, "eviction_rounds": 40},
            "predictive": {"hit_ratio": 0.85, "eviction_rounds": 42},
        },
    }
    r = build_evict_report(data)
    assert r["delta"]["hit_ratio"] == 0.05
    assert r["delta"]["eviction_rounds"] == 2


def test_render_text_contains_conclusion(tmp_path):
    p = tmp_path / "evict.json"
    import json

    p.write_text(json.dumps(SAMPLE), encoding="utf-8")
    data = load_evict_compare(p)
    r = build_evict_report(data)
    from kvscope.evict import render_text

    text = render_text(r)
    assert "LRU" in text
    assert "predictive" in text
    assert "fewer eviction rounds" in text


def test_render_text_empty_delta():
    data = {
        **SAMPLE,
        "strategies": {
            "lru": {"hit_ratio": 0.9, "eviction_rounds": 10},
            "predictive": {"hit_ratio": 0.9, "eviction_rounds": 10},
        },
    }
    r = build_evict_report(data)
    from kvscope.evict import render_text

    text = render_text(r)
    assert "equivalent" in text
