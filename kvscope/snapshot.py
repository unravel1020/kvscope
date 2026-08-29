# SPDX-License-Identifier: Apache-2.0
"""KVScope snapshot parsing.

Loads a radix-cache snapshot dump (JSONL) produced by a running SGLang
server (or by a simulator) into a normalized in-memory model.

Snapshot schema (v1), see docs/DESIGN.md:

    {"type": "snapshot", "version": 1, "page_size": 1, "eviction_policy": "lru",
     "pool": {"total_tokens": N, "used_tokens": N, ...},
     "nodes": [
       {"id": 0, "parent": -1, "token_len": 0, "children": [1], "lock_ref": 1,
        "hit_count": 0, "priority": 0, "evicted": false, "last_access": 0.0},
       ...
     ]}

The node list is flat and references parents/children by id, mirroring the
structure of ``sglang.srt.mem_cache.radix_cache.TreeNode`` without carrying
the token payloads.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SnapshotNode:
    """One node of the radix tree, mirroring ``TreeNode`` minus the tensor.

    ``token_len`` is ``len(node.key)`` in tokens (not bytes). ``evicted``
    corresponds to ``node.value is None`` in SGLang.
    """

    id: int
    parent: int
    token_len: int
    children: list[int] = field(default_factory=list)
    lock_ref: int = 0
    hit_count: int = 0
    priority: int = 0
    evicted: bool = False
    last_access: float = 0.0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SnapshotNode":
        required = {"id", "parent", "token_len"}
        missing = required - set(d)
        if missing:
            raise ValueError(f"node missing required field(s): {sorted(missing)}")
        return cls(
            id=int(d["id"]),
            parent=int(d["parent"]),
            token_len=int(d["token_len"]),
            children=[int(c) for c in d.get("children", [])],
            lock_ref=int(d.get("lock_ref", 0)),
            hit_count=int(d.get("hit_count", 0)),
            priority=int(d.get("priority", 0)),
            evicted=bool(d.get("evicted", False)),
            last_access=float(d.get("last_access", 0.0)),
        )


@dataclass
class PoolStats:
    total_tokens: int = 0
    used_tokens: int = 0
    evictable_tokens: int | None = None
    protected_tokens: int | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PoolStats":
        return cls(
            total_tokens=int(d.get("total_tokens", 0)),
            used_tokens=int(d.get("used_tokens", 0)),
            evictable_tokens=(
                int(d["evictable_tokens"]) if "evictable_tokens" in d else None
            ),
            protected_tokens=(
                int(d["protected_tokens"]) if "protected_tokens" in d else None
            ),
        )


@dataclass
class Snapshot:
    """A parsed radix-cache snapshot."""

    version: int
    page_size: int
    eviction_policy: str
    pool: PoolStats
    nodes: dict[int, SnapshotNode]  # id -> node

    @property
    def root_id(self) -> int:
        """The single node with parent == -1 (SGLang's root node)."""
        roots = [n.id for n in self.nodes.values() if n.parent == -1]
        if len(roots) != 1:
            raise ValueError(
                f"snapshot must have exactly one root (parent=-1), got {len(roots)}"
            )
        return roots[0]

    def total_tokens_in_tree(self) -> int:
        """Sum of token_len over all non-root nodes (root carries 0 tokens)."""
        return sum(n.token_len for n in self.nodes.values() if n.id != self.root_id)


def load_snapshot(path: str | Path) -> Snapshot:
    """Parse a snapshot JSONL file.

    The header record (``"type": "snapshot"``) carries version/page_size/
    eviction_policy/pool; every subsequent record is a node.
    """
    header: dict[str, Any] | None = None
    nodes: dict[int, SnapshotNode] = {}
    node_ids_seen: set[int] = set()

    with open(path, encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no + 1}: invalid JSON: {e}") from e

            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no + 1}: record is not an object")

            rtype = obj.get("type", "node")
            if rtype == "snapshot":
                if header is not None:
                    raise ValueError(f"{path}:{line_no + 1}: duplicate snapshot header")
                header = obj
                continue
            if rtype == "node":
                node = SnapshotNode.from_dict(obj)
                if node.id in node_ids_seen:
                    raise ValueError(
                        f"{path}:{line_no + 1}: duplicate node id {node.id}"
                    )
                node_ids_seen.add(node.id)
                nodes[node.id] = node
                continue
            raise ValueError(f"{path}:{line_no + 1}: unknown record type {rtype!r}")

    if header is None:
        raise ValueError(f"{path}: missing snapshot header record")

    version = int(header.get("version", 1))
    if version != 1:
        raise ValueError(f"unsupported snapshot version {version} (expected 1)")

    snapshot = Snapshot(
        version=version,
        page_size=int(header.get("page_size", 1)),
        eviction_policy=str(header.get("eviction_policy", "lru")),
        pool=PoolStats.from_dict(header.get("pool", {})),
        nodes=nodes,
    )

    # Structural validation: every referenced parent/child must exist.
    for node in snapshot.nodes.values():
        if node.parent != -1 and node.parent not in snapshot.nodes:
            raise ValueError(f"node {node.id} references missing parent {node.parent}")
        for child_id in node.children:
            if child_id not in snapshot.nodes:
                raise ValueError(
                    f"node {node.id} references missing child {child_id}"
                )

    # Exactly one root.
    snapshot.root_id  # raises if not exactly one

    return snapshot
