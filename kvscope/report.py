# SPDX-License-Identifier: Apache-2.0
"""Report assembly: combine snapshot header + tree metrics into the final
text/JSON report, mirroring vLLM's ``AnalysisReport.to_dict/render_text``
shape (flat summary + structured detail, dual output formats).
"""

from __future__ import annotations

from .snapshot import PoolStats, Snapshot
from .tree import TreeMetrics


def build_report(snapshot: Snapshot, metrics: TreeMetrics) -> dict:
    """Assemble the full report dict (shared by text and json renderers)."""
    return {
        "version": snapshot.version,
        "page_size": snapshot.page_size,
        "eviction_policy": snapshot.eviction_policy,
        "pool": _pool_to_dict(snapshot.pool),
        "metrics": metrics.to_dict(),
    }


def _pool_to_dict(pool: PoolStats) -> dict:
    d: dict = {
        "total_tokens": pool.total_tokens,
        "used_tokens": pool.used_tokens,
    }
    if pool.evictable_tokens is not None:
        d["evictable_tokens"] = pool.evictable_tokens
    if pool.protected_tokens is not None:
        d["protected_tokens"] = pool.protected_tokens
    return d


def render_text(report: dict) -> str:
    """Human-readable text rendering (the ``kvscope analyze`` default)."""
    m = report["metrics"]
    lines = [
        "SGLang KV Cache Analysis Report",
        "=" * 40,
        f"version             : {report['version']}",
        f"page size           : {report['page_size']}",
        f"eviction policy     : {report['eviction_policy']}",
        "",
        "[structure]",
        f"  nodes             : {m['structure']['num_nodes']}",
        f"  leaves / internal : {m['structure']['num_leaves']} / {m['structure']['num_internal']}",
        f"  depth max/median  : {m['structure']['max_depth']} / {m['structure']['median_depth']}",
        "[sharing]",
        f"  total tokens      : {m['sharing']['total_tokens']}",
        f"  unique tokens     : {m['sharing']['unique_tokens']}",
        f"  shared tokens     : {m['sharing']['shared_tokens']}",
        f"  reuse ratio       : {m['sharing']['reuse_ratio']:.2%}",
        f"  shared nodes      : {m['sharing']['num_shared_nodes']}",
        "[fragmentation]",
        f"  small nodes       : {m['fragmentation']['num_small_nodes']}",
        f"  avg token_len     : {m['fragmentation']['avg_token_len']}",
        "[eviction]",
        f"  evicted nodes     : {m['eviction']['num_evicted_nodes']} ({m['eviction']['evicted_tokens']} tokens)",
        f"  locked nodes      : {m['eviction']['num_locked_nodes']} ({m['eviction']['locked_tokens']} tokens)",
        "[hotness]",
        f"  max hit_count     : {m['hotness']['max_hit_count']}",
    ]
    hot = m["hotness"]["hot_nodes"]
    if hot:
        lines.append(
            "  top hit nodes      : "
            + ", ".join(f"#{nid}({hits})" for nid, hits in hot[:5])
        )
    return "\n".join(lines)
