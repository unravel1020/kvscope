# SPDX-License-Identifier: Apache-2.0
"""Gap-duration analysis and cache-decay prediction.

Motivation (TraceLab, arXiv 2606.30560): in real coding-agent workloads the
session spends ~92% of wall-clock time waiting for the human, and cache
reuse decays with the inter-turn gap: prefixes idle >5 minutes start to
miss, >1 hour they are almost always gone. A **predictive eviction**
strategy (cf. CacheWise) can use this signal: if the current gap predicts
low reuse probability for a prefix, evict it eagerly instead of keeping it
until LRU forces the issue.

This module:
1. Builds a gap histogram (the distribution of human idle time).
2. Measures gap -> next-turn hit ratio (the empirical decay curve).
3. Fits a simple logistic model P(hit | gap) and exposes
   ``cache_decay_probability(gap)`` — the basis for predictive eviction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .turns import TurnRecord, load_turns

# Empirical anchor from TraceLab: >5 min the cache starts to miss.
DEFAULT_DECAY_THRESHOLD_S = 300.0


@dataclass
class GapStats:
    """Gap distribution + decay relationship."""

    num_turns: int
    gaps: list[float]
    gap_histogram: list[tuple[str, int]]  # (bucket label, count)
    decay_bins: list[dict[str, Any]]  # per-gap-bin: hit ratio + n
    decay_threshold_s: float
    # Logistic model params: P(hit) = 1 / (1 + exp(-(a + b*log(gap))))
    logistic_a: float
    logistic_b: float

    def predict_hit_probability(self, gap_seconds: float) -> float:
        """Estimated P(next turn hits cache) given an idle gap."""
        if gap_seconds <= 0:
            return 1.0
        z = self.logistic_a + self.logistic_b * math.log(gap_seconds)
        return 1.0 / (1.0 + math.exp(-z))

    def cache_decay_probability(self, gap_seconds: float) -> float:
        """P(cache for this prefix is gone / useless) = 1 - P(hit)."""
        return 1.0 - self.predict_hit_probability(gap_seconds)

    def to_dict(self) -> dict:
        return {
            "num_turns": self.num_turns,
            "decay_threshold_s": self.decay_threshold_s,
            "gap_histogram": self.gap_histogram,
            "decay_bins": self.decay_bins,
            "logistic_model": {
                "a": round(self.logistic_a, 4),
                "b": round(self.logistic_b, 4),
            },
            "anchor_predictions": {
                "60s": round(self.predict_hit_probability(60), 4),
                "300s": round(self.predict_hit_probability(300), 4),
                "1800s": round(self.predict_hit_probability(1800), 4),
                "3600s": round(self.predict_hit_probability(3600), 4),
            },
        }

    def render_text(self) -> str:
        lines = [
            "Gap-Duration & Cache Decay Report",
            "=" * 40,
            f"turns analyzed        : {self.num_turns}",
            f"decay threshold       : {self.decay_threshold_s:.0f}s",
            "",
            "[gap histogram]",
        ]
        for label, count in self.gap_histogram:
            bar = "#" * int(count / max(max((c for _, c in self.gap_histogram), default=1), 1) * 30)
            lines.append(f"  {label:>10}: {count:>4} |{bar}")
        lines.append("")
        lines.append("[gap -> next-turn hit ratio (empirical decay)]")
        for b in self.decay_bins:
            bar = "#" * int(b["hit_ratio"] * 30)
            lines.append(
                f"  {b['label']:>10}: {b['hit_ratio']:6.1%} (n={b['n']}) |{bar}"
            )
        lines.append("")
        lines.append("[logistic model: P(hit | gap)]")
        lines.append(f"  model   : P = 1/(1+exp(-(a + b*ln(gap))))")
        lines.append(f"  a, b    : {self.logistic_a:.3f}, {self.logistic_b:.3f}")
        lines.append(f"  @ 60s   : {self.predict_hit_probability(60):.1%}")
        lines.append(f"  @ 5min  : {self.predict_hit_probability(300):.1%}")
        lines.append(f"  @ 30min : {self.predict_hit_probability(1800):.1%}")
        lines.append(f"  @ 1h    : {self.predict_hit_probability(3600):.1%}")
        lines.append("")
        lines.append("[predictive-eviction signal: cache_decay_probability(gap)]")
        lines.append(f"  @ 5min  : {self.cache_decay_probability(300):.1%}  (evict candidates)")
        return "\n".join(lines)


_GAP_BUCKETS = [
    ("<10s", 0, 10),
    ("10-60s", 10, 60),
    ("1-5min", 60, 300),
    ("5-30min", 300, 1800),
    (">30min", 1800, float("inf")),
]


def _bucket_label(gap: float) -> str:
    for label, lo, hi in _GAP_BUCKETS:
        if lo <= gap < hi:
            return label
    return ">30min"


def analyze_gaps(
    records: list[TurnRecord],
    *,
    decay_threshold_s: float = DEFAULT_DECAY_THRESHOLD_S,
) -> GapStats:
    """Compute gap statistics and fit the decay model.

    The logistic fit uses least-squares on z = ln(P/(1-P)) vs ln(gap),
    restricted to turns with 0 < gap (a gap of 0 is the first turn or an
    immediate follow-up and carries no decay signal).
    """
    # Gap distribution (all turns except the very first).
    gaps = [r.gap_seconds for r in records if r.gap_seconds > 0]

    # Histogram.
    hist: dict[str, int] = {}
    for g in gaps:
        lbl = _bucket_label(g)
        hist[lbl] = hist.get(lbl, 0) + 1
    gap_histogram = [(lbl, hist.get(lbl, 0)) for lbl, _, _ in _GAP_BUCKETS]

    # Decay bins: for each gap bucket, next-turn hit ratio. We pair each
    # turn's gap with ITS hit ratio (the gap preceding this turn tells us
    # how likely this turn's prefix survived).
    bin_hits: dict[str, list[float]] = {}
    for r in records:
        if r.gap_seconds <= 0:
            continue
        lbl = _bucket_label(r.gap_seconds)
        bin_hits.setdefault(lbl, []).append(r.hit_ratio)
    decay_bins: list[dict[str, Any]] = []
    for lbl, _, _ in _GAP_BUCKETS:
        hits = bin_hits.get(lbl, [])
        decay_bins.append(
            {
                "label": lbl,
                "hit_ratio": (sum(hits) / len(hits)) if hits else 0.0,
                "n": len(hits),
            }
        )

    # Logistic fit on log-gap vs hit (0/1 softened by hit_ratio).
    xs: list[float] = []
    ys: list[float] = []
    for r in records:
        if 0 < r.gap_seconds:
            xs.append(math.log(r.gap_seconds))
            ys.append(r.hit_ratio)
    a, b = _fit_logistic(xs, ys)

    return GapStats(
        num_turns=len(records),
        gaps=gaps,
        gap_histogram=gap_histogram,
        decay_bins=decay_bins,
        decay_threshold_s=decay_threshold_s,
        logistic_a=a,
        logistic_b=b,
    )


def _fit_logistic(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least-squares fit of z = a + b*x where z = ln(P/(1-P)).

    Falls back to (a=2.0, b=-0.35) when there are too few points or the
    values are degenerate (all hits or all misses).
    """
    pts = [(x, y) for x, y in zip(xs, ys) if 0.0 < y < 1.0]
    if len(pts) < 3:
        return 2.0, -0.35  # soft default: mild decay
    n = len(pts)
    sx = sum(p[0] for p in pts)
    sy = sum(math.log(p[1] / (1 - p[1])) for p in pts)
    sxx = sum(p[0] * p[0] for p in pts)
    sxy = sum(p[0] * math.log(p[1] / (1 - p[1])) for p in pts)
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return 2.0, -0.35
    b = (n * sxy - sx * sy) / denom
    a = (sy - b * sx) / n
    if not math.isfinite(a) or not math.isfinite(b):
        return 2.0, -0.35
    return a, b


def analyze_gaps_from_file(
    path: str | Path, *, decay_threshold_s: float = DEFAULT_DECAY_THRESHOLD_S
) -> GapStats:
    """Load turns JSONL and run gap analysis."""
    records = load_turns(path)
    return analyze_gaps(records, decay_threshold_s=decay_threshold_s)
