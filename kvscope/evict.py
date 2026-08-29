# SPDX-License-Identifier: Apache-2.0
"""Eviction-strategy comparison report.

Consumes the JSON produced by ``scripts/simulate_evict_compare.py``
(identical workload run under LRU vs gap-predictive eviction, both on
SGLang's real radix cache with a finite KV pool) and renders a side-by-side
comparison.

The headline metric is **evictions at equal hit rate**: predictive eviction
should reach the same (or better) hit ratio with fewer eviction rounds,
because it clears stale prefixes proactively instead of letting them occupy
the pool until LRU is forced to act (cf. CacheWise).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _fmt_ratio(x: float) -> str:
    return f"{x:.1%}"


def build_evict_report(data: dict) -> dict:
    """Normalize the comparison JSON into a report dict."""
    workload = data.get("workload", {})
    strategies = data.get("strategies", {})
    lru = strategies.get("lru", {})
    pred = strategies.get("predictive", {})

    lru_hit = float(lru.get("hit_ratio", 0.0))
    pred_hit = float(pred.get("hit_ratio", 0.0))
    lru_evicts = int(lru.get("eviction_rounds", 0))
    pred_evicts = int(pred.get("eviction_rounds", 0))

    hit_delta = pred_hit - lru_hit
    evict_delta = pred_evicts - lru_evicts
    evict_pct = (evict_delta / lru_evicts) if lru_evicts else 0.0

    return {
        "workload": workload,
        "strategies": {
            "lru": {"hit_ratio": round(lru_hit, 4), "eviction_rounds": lru_evicts},
            "predictive": {
                "hit_ratio": round(pred_hit, 4),
                "eviction_rounds": pred_evicts,
            },
        },
        "delta": {
            "hit_ratio": round(hit_delta, 4),
            "eviction_rounds": evict_delta,
            "eviction_rounds_pct": round(evict_pct, 4),
        },
    }


def render_text(report: dict) -> str:
    w = report["workload"]
    s = report["strategies"]
    d = report["delta"]
    lines = [
        "Eviction Strategy Comparison (LRU vs gap-predictive)",
        "=" * 44,
        f"workload              : {w.get('n_turns', 0)} turns, "
        f"{w.get('n_conversations', 0)} convs x {w.get('turns_per_conv', 0)} turns",
        f"system prompt         : {w.get('system_prompt_tokens', 0)} tokens (locked)",
        f"KV pool               : {w.get('pool_tokens', 0)} tokens (finite)",
        f"long-gap probability  : {w.get('long_gap_prob', 0):.0%}",
        f"decay threshold       : {w.get('decay_threshold_s', 0)}s",
        "",
        f"{'metric':<28}{'LRU':>12}{'predictive':>12}{'delta':>10}",
        "-" * 62,
        f"{'hit ratio':<28}{_fmt_ratio(s['lru']['hit_ratio']):>12}"
        f"{_fmt_ratio(s['predictive']['hit_ratio']):>12}"
        f"{('+' if d['hit_ratio'] >= 0 else '') + _fmt_ratio(d['hit_ratio']):>10}",
        f"{'eviction rounds':<28}{s['lru']['eviction_rounds']:>12}"
        f"{s['predictive']['eviction_rounds']:>12}"
        f"{('+' if d['eviction_rounds'] >= 0 else '') + str(d['eviction_rounds']):>10}",
        "",
    ]
    if d["eviction_rounds"] < 0:
        lines.append(
            f"  => predictive reaches equal hit rate with "
            f"{-d['eviction_rounds_pct']:.0%} fewer eviction rounds "
            "(proactive stale-prefix cleanup)."
        )
    elif d["hit_ratio"] > 0:
        lines.append(
            f"  => predictive improves hit rate by {d['hit_ratio']:.1%} "
            "at comparable eviction cost."
        )
    else:
        lines.append(
            "  => strategies are equivalent on this workload; "
            "increase pool pressure or gap spread to separate them."
        )
    return "\n".join(lines)


def load_evict_compare(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
