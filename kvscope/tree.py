# SPDX-License-Identifier: Apache-2.0
"""Radix-tree analysis over a parsed snapshot.

Computes structural/sharing/fragmentation/eviction/hotness metrics with a
single DFS over the flat node list, in the spirit of vLLM's
``_group_requests_by_shared_prefix`` (one O(total nodes) pass, no quadratic
path materialization). Every metric here is derived from the snapshot
structure only — no token payloads required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .snapshot import Snapshot, SnapshotNode


@dataclass
class TreeMetrics:
    """All derived metrics for one snapshot."""

    # Structure
    num_nodes: int = 0
    max_depth: int = 0
    median_depth: int = 0
    depth_histogram: dict[int, int] = field(default_factory=dict)  # depth -> count
    num_leaves: int = 0
    num_internal: int = 0

    # Sharing
    total_tokens: int = 0  # sum of token_len over non-root nodes
    unique_tokens: int = 0  # tokens that appear in exactly one node (leaf-only path)
    shared_tokens: int = 0  # tokens in nodes with >1 child (branching = shared prefix)
    num_shared_nodes: int = 0  # nodes with >1 child
    reuse_ratio: float = 0.0  # 1 - unique/total (0..1)

    # Fragmentation
    num_split_nodes: int = 0  # nodes with token_len < some threshold (small = split residue)
    num_small_nodes: int = 0  # token_len < 16 (page-size-ish heuristic)
    avg_token_len: float = 0.0

    # Eviction
    num_evicted_nodes: int = 0
    evicted_tokens: int = 0
    num_locked_nodes: int = 0  # lock_ref > 0
    locked_tokens: int = 0

    # Hotness
    hot_nodes: list[tuple[int, int]] = field(default_factory=list)  # (id, hit_count)
    max_hit_count: int = 0

    def to_dict(self) -> dict:
        return {
            "structure": {
                "num_nodes": self.num_nodes,
                "num_leaves": self.num_leaves,
                "num_internal": self.num_internal,
                "max_depth": self.max_depth,
                "median_depth": self.median_depth,
                "depth_histogram": dict(sorted(self.depth_histogram.items())),
            },
            "sharing": {
                "total_tokens": self.total_tokens,
                "unique_tokens": self.unique_tokens,
                "shared_tokens": self.shared_tokens,
                "num_shared_nodes": self.num_shared_nodes,
                "reuse_ratio": round(self.reuse_ratio, 4),
            },
            "fragmentation": {
                "num_split_nodes": self.num_split_nodes,
                "num_small_nodes": self.num_small_nodes,
                "avg_token_len": round(self.avg_token_len, 2),
            },
            "eviction": {
                "num_evicted_nodes": self.num_evicted_nodes,
                "evicted_tokens": self.evicted_tokens,
                "num_locked_nodes": self.num_locked_nodes,
                "locked_tokens": self.locked_tokens,
            },
            "hotness": {
                "max_hit_count": self.max_hit_count,
                "hot_nodes": self.hot_nodes[:10],
            },
        }

    def render_text(self) -> str:
        lines = [
            "Radix Cache Analysis Report",
            "=" * 40,
            f"nodes                 : {self.num_nodes} (leaves {self.num_leaves}, internal {self.num_internal})",
            f"depth                 : max {self.max_depth}, median {self.median_depth}",
            f"tokens                : {self.total_tokens} (unique {self.unique_tokens})",
            f"reuse ratio           : {self.reuse_ratio:.2%}",
            f"shared nodes (>1 child): {self.num_shared_nodes} ({self.num_shared_nodes / max(self.num_nodes, 1):.1%})",
            f"shared tokens         : {self.shared_tokens}",
            f"small nodes (<16 tok) : {self.num_small_nodes} ({self.num_small_nodes / max(self.num_nodes, 1):.1%})",
            f"avg token_len         : {self.avg_token_len:.1f}",
            f"evicted nodes         : {self.num_evicted_nodes} ({self.evicted_tokens} tokens)",
            f"locked nodes          : {self.num_locked_nodes} ({self.locked_tokens} tokens)",
            f"max hit_count         : {self.max_hit_count}",
        ]
        if self.hot_nodes:
            lines.append("top hit nodes          : " + ", ".join(
                f"#{nid}({hits})" for nid, hits in self.hot_nodes[:5]
            ))
        return "\n".join(lines)


_SMALL_NODE_THRESHOLD = 16


def analyze_tree(
    snapshot: Snapshot,
    *,
    small_node_threshold: int = _SMALL_NODE_THRESHOLD,
    split_node_threshold: int = _SMALL_NODE_THRESHOLD,
    top_hot: int = 10,
) -> TreeMetrics:
    """Run the analysis DFS. ``small``/``split`` thresholds are heuristic
    fragmentation markers (a node shorter than one page is usually split
    residue from ``_split_node`` partial matches).
    """
    m = TreeMetrics()
    m.num_nodes = len(snapshot.nodes)
    root_id = snapshot.root_id
    root = snapshot.nodes[root_id]

    # We compute per-node metrics with an explicit stack carrying depth.
    # unique_tokens is computed as: tokens in nodes that are NOT a shared
    # prefix (i.e. nodes whose subtree never branches and which have no
    # sibling sharing above). Simpler robust definition used here:
    #   unique = sum of token_len over nodes with exactly 0 children that are
    #            not reachable via any branch node... but that double counts.
    # We instead count reuse directly: a token is "reused" iff its node has a
    # sibling somewhere in its ancestry chain that also covers it — i.e. the
    # node lies under a branching node. So:
    #   unique_tokens = tokens not under any branching node
    #   shared_tokens = tokens under at least one branching node
    # Both sum to total_tokens. This is the token-level analogue of vLLM's
    # "count reuse once per distinct block-hash" (no double counting: each
    # node's tokens are counted exactly once, in one bucket).

    stack: list[tuple[SnapshotNode, int, bool]] = [
        (root, 0, False)  # (node, depth, under_branch)
    ]
    depths: list[int] = []
    total = 0
    unique = 0
    shared = 0

    while stack:
        node, depth, under_branch = stack.pop()
        if depth > 0:
            depths.append(depth)
            total += node.token_len
            if under_branch:
                shared += node.token_len
            else:
                unique += node.token_len

        m.depth_histogram[depth] = m.depth_histogram.get(depth, 0) + 1
        m.max_depth = max(m.max_depth, depth)

        children = [snapshot.nodes[c] for c in node.children if c in snapshot.nodes]
        is_branch = len(children) > 1
        if is_branch:
            m.num_shared_nodes += 1
        if depth > 0:
            m.num_internal += 1 if children else 0
            m.num_leaves += 0 if children else 1
            if node.lock_ref > 0:
                m.num_locked_nodes += 1
                m.locked_tokens += node.token_len
            if node.evicted:
                m.num_evicted_nodes += 1
                m.evicted_tokens += node.token_len
            if node.token_len < split_node_threshold:
                m.num_split_nodes += 1
            if node.token_len < small_node_threshold:
                m.num_small_nodes += 1
            if node.hit_count > 0:
                m.hot_nodes.append((node.id, node.hit_count))
                m.max_hit_count = max(m.max_hit_count, node.hit_count)

        child_under_branch = under_branch or is_branch
        for child in children:
            stack.append((child, depth + 1, child_under_branch))

    m.total_tokens = total
    m.unique_tokens = unique
    m.shared_tokens = shared
    m.reuse_ratio = (shared / total) if total > 0 else 0.0
    m.avg_token_len = (total / max(len(depths), 1)) if depths else 0.0
    m.median_depth = _median(depths)
    m.hot_nodes.sort(key=lambda x: (-x[1], x[0]))
    m.hot_nodes = m.hot_nodes[:top_hot]

    # Sanity: unique + shared must equal total.
    assert unique + shared == total, (unique, shared, total)
    return m


def _median(values: list[int]) -> int:
    if not values:
        return 0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) // 2
