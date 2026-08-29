# SPDX-License-Identifier: Apache-2.0
"""Tests for dump.py: serializing a radix-tree-like object to snapshot JSONL.

Uses a minimal fake tree (no SGLang dependency), mirroring the TreeNode
shape SGLang exposes (children dict, key with __len__, lock_ref, hit_count,
priority, evicted, id, parent).
"""

from __future__ import annotations

import json

from kvscope.dump import dump_radix_cache, write_snapshot


class FakeKey:
    def __init__(self, n: int):
        self._n = n

    def __len__(self):
        return self._n


class FakeNode:
    _counter = 0

    def __init__(self, token_len=0, parent=None, children=None, **kw):
        self.id = FakeNode._counter
        FakeNode._counter += 1
        self.key = FakeKey(token_len) if token_len > 0 else None
        self.parent = parent
        self.children = {f"k{i}": c for i, c in enumerate(children or [])}
        self.lock_ref = kw.get("lock_ref", 0)
        self.hit_count = kw.get("hit_count", 0)
        self.priority = kw.get("priority", 0)
        self.evicted = kw.get("evicted", False)
        self.last_access_time = kw.get("last_access_time", 0.0)
        self.value = None if self.evicted else object()

    @property
    def evicted(self):
        return self._evicted

    @evicted.setter
    def evicted(self, v):
        self._evicted = v


class FakeCache:
    """Mimics the RadixCache surface dump.py reads (root_node + metadata)."""

    def __init__(self, root):
        self.root_node = root
        self.eviction_policy = "lru"
        self.page_size = 1


def _make_tree():
    """root -> n1(100) -> {n2(40), n3(60, evicted)}"""
    FakeNode._counter = 0
    n2 = FakeNode(token_len=40, evicted=False, hit_count=3)
    n3 = FakeNode(token_len=60, evicted=True)
    n1 = FakeNode(token_len=100, children=[n2, n3], lock_ref=1, hit_count=9)
    root = FakeNode(token_len=0, children=[n1])
    return FakeCache(root)


def test_dump_header_and_nodes():
    cache = _make_tree()
    records = dump_radix_cache(cache, eviction_policy="lru", page_size=1)
    # header + 4 nodes = 5
    assert len(records) == 5

    header = records[0]
    assert header["type"] == "snapshot"
    assert header["version"] == 1
    assert header["eviction_policy"] == "lru"
    assert header["page_size"] == 1

    nodes = records[1:]
    # Find nodes by (token_len, hit_count) instead of assuming ids.
    n100 = next(n for n in nodes if n["token_len"] == 100)
    n40 = next(n for n in nodes if n["token_len"] == 40)
    n60 = next(n for n in nodes if n["token_len"] == 60)
    root_rec = next(n for n in nodes if n["token_len"] == 0)

    assert n100["lock_ref"] == 1
    assert n100["hit_count"] == 9
    assert n100["children"] == [n40["id"], n60["id"]]
    assert n100["parent"] == root_rec["id"]
    assert n40["evicted"] is False
    assert n60["evicted"] is True


def test_write_snapshot_jsonl_roundtrip():
    cache = _make_tree()
    import io

    buf = io.StringIO()
    n = write_snapshot(cache, buf, page_size=2)
    lines = [json.loads(l) for l in buf.getvalue().strip().splitlines()]
    assert n == len(lines) == 5
    assert lines[0]["page_size"] == 2
    assert lines[-1]["token_len"] == 60
