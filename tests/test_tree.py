# SPDX-License-Identifier: Apache-2.0
"""Tests for snapshot parsing and tree analysis, using synthetic snapshots
(no tokenizer, no SGLang dependency) — same strategy as vLLM's
``test_prefix_cache_analysis`` regression tests.
"""

from __future__ import annotations

import json

import pytest

from kvscope.snapshot import Snapshot, SnapshotNode, load_snapshot
from kvscope.tree import analyze_tree


def _node(id, parent, token_len, children=None, **kw):
    return {"id": id, "parent": parent, "token_len": token_len,
            "children": children or [], **kw}


def _header(**kw):
    d = {"type": "snapshot", "version": 1, "page_size": 1,
         "eviction_policy": "lru", "pool": {"total_tokens": 0, "used_tokens": 0}}
    d.update(kw)
    return d


def _write(tmp_path, records):
    p = tmp_path / "snapshot.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return p


# ---------- snapshot loading ----------

def test_load_basic_snapshot(tmp_path):
    p = _write(tmp_path, [
        _header(pool={"total_tokens": 1000, "used_tokens": 600}),
        _node(0, -1, 0, children=[1]),
        _node(1, 0, 100, children=[2, 3], hit_count=8, lock_ref=2),
        _node(2, 1, 40, evicted=True),
        _node(3, 1, 60),
    ])
    snap = load_snapshot(p)
    assert snap.version == 1
    assert snap.page_size == 1
    assert snap.pool.used_tokens == 600
    assert snap.root_id == 0
    assert snap.total_tokens_in_tree() == 200
    assert snap.nodes[1].hit_count == 8
    assert snap.nodes[2].evicted is True


def test_load_missing_prompt_field_like_error(tmp_path):
    # A node without the required id field must fail.
    p = _write(tmp_path, [
        _header(),
        {"parent": -1, "token_len": 0},  # missing id
    ])
    with pytest.raises(ValueError, match="missing required field"):
        load_snapshot(p)


def test_load_missing_header(tmp_path):
    p = _write(tmp_path, [_node(0, -1, 0)])
    with pytest.raises(ValueError, match="missing snapshot header"):
        load_snapshot(p)


def test_load_duplicate_node_id(tmp_path):
    p = _write(tmp_path, [
        _header(),
        _node(0, -1, 0),
        _node(0, -1, 5),
    ])
    with pytest.raises(ValueError, match="duplicate node id"):
        load_snapshot(p)


def test_load_dangling_parent(tmp_path):
    p = _write(tmp_path, [
        _header(),
        _node(0, -1, 0, children=[1]),
        # node 1 never defined
    ])
    with pytest.raises(ValueError, match="missing child"):
        load_snapshot(p)


def test_load_must_have_single_root(tmp_path):
    p = _write(tmp_path, [
        _header(),
        _node(0, -1, 0),
        _node(1, -1, 10),
    ])
    with pytest.raises(ValueError, match="exactly one root"):
        load_snapshot(p)


def test_load_blank_lines_ignored(tmp_path):
    p = _write(tmp_path, [
        _header(),
        _node(0, -1, 0, children=[1]),
        _node(1, 0, 5),
    ])
    # _write already ends with \n; add an extra blank line via write
    p.write_text(p.read_text() + "\n\n", encoding="utf-8")
    snap = load_snapshot(p)
    assert snap.root_id == 0


# ---------- tree analysis ----------

def _chain_snapshot(nodes: list[dict]) -> Snapshot:
    """Build a Snapshot object directly (no file round-trip)."""
    parsed = {n["id"]: SnapshotNode.from_dict(n) for n in nodes}
    header = _header()
    return Snapshot(
        version=header["version"],
        page_size=header["page_size"],
        eviction_policy=header["eviction_policy"],
        pool=type("P", (), {"total_tokens": 0, "used_tokens": 0})(),
        nodes=parsed,
    )


def test_analyze_linear_chain_no_sharing():
    # 0 -> 1(50) -> 2(30): no branching, all tokens unique.
    snap = _chain_snapshot([
        _node(0, -1, 0, children=[1]),
        _node(1, 0, 50, children=[2]),
        _node(2, 1, 30),
    ])
    m = analyze_tree(snap)
    assert m.total_tokens == 80
    assert m.unique_tokens == 80
    assert m.shared_tokens == 0
    assert m.reuse_ratio == 0.0
    assert m.num_shared_nodes == 0
    assert m.max_depth == 2
    assert m.num_leaves == 1


def test_analyze_branch_sharing():
    # 0 -> 1(100) -> {2(50), 3(50)}: node 1's 100 tokens are a shared prefix.
    snap = _chain_snapshot([
        _node(0, -1, 0, children=[1]),
        _node(1, 0, 100, children=[2, 3]),
        _node(2, 1, 50),
        _node(3, 1, 50),
    ])
    m = analyze_tree(snap)
    assert m.total_tokens == 200
    assert m.shared_tokens == 100  # node 1 under a branch
    assert m.unique_tokens == 100  # leaves 2,3 not under branch
    assert m.reuse_ratio == 0.5
    assert m.num_shared_nodes == 1
    assert m.num_leaves == 2


def test_analyze_nested_branch_counts_once():
    # Shared tokens must be counted once even under nested branches
    # (analogue of vLLM's double-counting regression test).
    # 0 -> 1(10) -> {2(5), 3(5)} ; 2 -> 4(3)
    #
    # Semantics: "shared tokens" = tokens under at least one branching node.
    #   node 1 is the branch itself (its own 10 tokens are NOT under a branch
    #   node, they are the shared carrier), so it counts as unique.
    #   nodes 2,3,4 are under branch node 1 -> shared.
    #   shared = 5 + 5 + 3 = 13 ; unique = 10 ; total = 23
    snap = _chain_snapshot([
        _node(0, -1, 0, children=[1]),
        _node(1, 0, 10, children=[2, 3]),
        _node(2, 1, 5, children=[4]),
        _node(3, 1, 5),
        _node(4, 2, 3),
    ])
    m = analyze_tree(snap)
    assert m.total_tokens == 23
    assert m.unique_tokens == 10
    assert m.shared_tokens == 13
    assert m.reuse_ratio == 13 / 23


def test_analyze_evicted_and_locked():
    snap = _chain_snapshot([
        _node(0, -1, 0, children=[1, 2]),
        _node(1, 0, 10, evicted=True),
        _node(2, 0, 20, lock_ref=3),
    ])
    m = analyze_tree(snap)
    assert m.num_evicted_nodes == 1
    assert m.evicted_tokens == 10
    assert m.num_locked_nodes == 1
    assert m.locked_tokens == 20


def test_analyze_hot_nodes():
    snap = _chain_snapshot([
        _node(0, -1, 0, children=[1, 2]),
        _node(1, 0, 10, hit_count=99),
        _node(2, 0, 20, hit_count=5),
    ])
    m = analyze_tree(snap)
    assert m.max_hit_count == 99
    assert m.hot_nodes[0] == (1, 99)
    assert m.hot_nodes[1] == (2, 5)


def test_analyze_small_nodes_fragmentation():
    snap = _chain_snapshot([
        _node(0, -1, 0, children=[2]),
        _node(2, 0, 40, children=[3]),
        _node(3, 2, 3),
    ])
    m = analyze_tree(snap, small_node_threshold=16)
    assert m.num_small_nodes == 1  # node 3 (3 tokens < 16)
