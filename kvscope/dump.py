# SPDX-License-Identifier: Apache-2.0
"""Dump a running SGLang radix tree to the KVScope snapshot format.

``dump_radix_cache`` walks a ``sglang.srt.mem_cache.radix_cache.RadixCache``
instance (or anything exposing the same TreeNode shape: ``children``,
``key``/``token_len``, ``lock_ref``, ``hit_count``, ``priority``, ``value``)
and emits the v1 snapshot JSONL consumed by ``kvscope analyze``.

Design notes:
- This module is import-safe without SGLang installed: SGLang types are
  only used inside ``TYPE_CHECKING``. The walker is duck-typed.
- ``token_len`` = ``len(node.key)`` (radix tokens), mirroring SGLang's
  ``TreeNode.key`` semantics; the root node carries 0 tokens.
- ``evicted`` = ``node.value is None`` (SGLang's ``TreeNode.evicted``).
- Node ids are the SGLang ``TreeNode.id`` values, preserving identity
  across a dump (parent/children reference these ids).
"""

from __future__ import annotations

import json
from typing import Any, TextIO

# SGLang is only imported for type hints; runtime uses duck typing.
try:  # pragma: no cover - optional import
    from sglang.srt.mem_cache.radix_cache import RadixCache  # noqa: F401
    from sglang.srt.mem_cache.radix_cache import TreeNode  # noqa: F401

    HAS_SGLANG = True
except ImportError:  # pragma: no cover
    HAS_SGLANG = False


def _token_len(node: Any) -> int:
    """Number of radix tokens carried by a node (root = 0)."""
    key = getattr(node, "key", None)
    if key is None:
        return 0
    try:
        return len(key)
    except TypeError:
        # Some cache variants store a raw array/tuple as key.
        return len(key)


def dump_radix_cache(
    cache: Any,
    *,
    eviction_policy: str | None = None,
    page_size: int | None = None,
    extra_pool: dict | None = None,
) -> list[dict]:
    """Serialize a radix cache into the v1 snapshot node list.

    Args:
        cache: a RadixCache-like object (root_node with children/keys).
        eviction_policy: optional override (defaults to cache.eviction_policy).
        page_size: optional override (defaults to cache.page_size).
        extra_pool: optional extra pool stats merged into the header.

    Returns:
        List of JSON-serializable records: the header dict first, then one
        node dict per TreeNode (DFS pre-order).
    """
    policy = eviction_policy or getattr(cache, "eviction_policy", "lru")
    page = page_size if page_size is not None else getattr(cache, "page_size", 1)

    header: dict[str, Any] = {
        "type": "snapshot",
        "version": 1,
        "page_size": int(page),
        "eviction_policy": str(policy),
        "pool": {
            "total_tokens": 0,
            "used_tokens": 0,
        },
    }
    pool = getattr(cache, "_pool", None)
    if pool is not None:  # pragma: no cover - depends on cache variant
        header["pool"].update(pool)
    if extra_pool:
        header["pool"].update(extra_pool)

    records: list[dict] = [header]

    root = getattr(cache, "root_node", None)
    if root is None:  # pragma: no cover - defensive
        return records

    # DFS pre-order, tracking depth for the node dict.
    stack: list[tuple[Any, int]] = [(root, 0)]
    while stack:
        node, depth = stack.pop()

        children = list(getattr(node, "children", {}).values())
        rec: dict[str, Any] = {
            "id": int(getattr(node, "id", 0)),
            "parent": -1 if depth == 0 else _find_parent_id(cache, node),
            "token_len": _token_len(node),
            "children": [int(c.id) for c in children],
            "lock_ref": int(getattr(node, "lock_ref", 0)),
            "hit_count": int(getattr(node, "hit_count", 0)),
            "priority": int(getattr(node, "priority", 0)),
            "evicted": bool(getattr(node, "evicted", False)),
            "last_access": float(getattr(node, "last_access_time", 0.0)),
        }
        records.append(rec)

        # Push children in reverse so DFS order matches insertion order.
        for child in reversed(children):
            stack.append((child, depth + 1))

    return records


def _find_parent_id(cache: Any, node: Any) -> int:
    """Resolve a node's parent id.

    SGLang TreeNode.parent is a back-reference; use it when available,
    otherwise fall back to a reverse lookup (O(n) — only used for caches
    without parent pointers, e.g. some C++ radix bindings).
    """
    parent = getattr(node, "parent", None)
    if parent is not None:
        return int(parent.id)
    # Fallback: reverse search (rarely needed).
    root = getattr(cache, "root_node", None)
    if root is None:
        return -1
    for candidate in _iter_nodes(root):
        if node in list(getattr(candidate, "children", {}).values()):
            return int(candidate.id)
    return -1


def _iter_nodes(node: Any):
    stack = [node]
    while stack:
        cur = stack.pop()
        yield cur
        for child in getattr(cur, "children", {}).values():
            stack.append(child)


def write_snapshot(cache: Any, out: TextIO, **kwargs: Any) -> int:
    """Dump a radix cache to an open text stream as snapshot JSONL.

    Returns the number of records written (header + nodes).
    """
    records = dump_radix_cache(cache, **kwargs)
    for rec in records:
        out.write(json.dumps(rec) + "\n")
    return len(records)
