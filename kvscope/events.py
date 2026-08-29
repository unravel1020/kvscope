# SPDX-License-Identifier: Apache-2.0
"""KV cache event-stream analysis.

SGLang's ``KVCacheEventRecorder`` produces a placement event stream
(``BlockStored`` / ``BlockRemoved`` / ``AllBlocksCleared``) consumed by
KV-aware routers. KVScope consumes the *same* events offline, giving the
dynamic view a static snapshot cannot: cache lifecycle, eviction timeline,
per-medium (GPU/CPU/DISK) movement, and churn rate.

Event schema (v1, mirrors SGLang's kv_events.py):

    {"type": "stored",  "block_hashes": [int, ...], "parent_block_hash": int|null,
     "token_ids": [int, ...], "block_size": int, "lora_id": int|null, "medium": "GPU"}
    {"type": "removed", "block_hashes": [int, ...], "medium": "GPU"}
    {"type": "cleared"}
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EventRecord:
    kind: str  # "stored" | "removed" | "cleared"
    block_hashes: list[int] = field(default_factory=list)
    parent_block_hash: int | None = None
    num_tokens: int = 0  # sum of block sizes for stored events
    medium: str = "GPU"
    seq: int = 0  # global sequence number (assigned on load)

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"type": self.kind, "medium": self.medium}
        if self.block_hashes:
            d["block_hashes"] = self.block_hashes
        if self.kind == "stored":
            d["parent_block_hash"] = self.parent_block_hash
            d["num_tokens"] = self.num_tokens
        return d


@dataclass
class EventStream:
    """Parsed event stream with derived metrics."""

    events: list[EventRecord]
    num_stored: int = 0
    num_removed: int = 0
    num_cleared: int = 0
    total_stored_tokens: int = 0
    total_removed_tokens: int = 0
    # eviction timeline: (seq, block_hash, medium) for each removal
    removal_timeline: list[tuple[int, int, str]] = field(default_factory=list)
    # per-medium token movement
    medium_stored_tokens: dict[str, int] = field(default_factory=dict)
    medium_removed_tokens: dict[str, int] = field(default_factory=dict)
    # store->remove pairing: how many stored blocks ever got removed
    blocks_stored: set[int] = field(default_factory=set)
    blocks_removed: set[int] = field(default_factory=set)
    # parent chain depth hints: count of chained stores (block_hashes sharing parents)
    chained_stores: int = 0

    @property
    def churn_rate(self) -> float:
        """Fraction of stored blocks that were later removed (0..1)."""
        if not self.blocks_stored:
            return 0.0
        return len(self.blocks_removed & self.blocks_stored) / len(self.blocks_stored)

    @property
    def reuse_confirmed(self) -> int:
        """Blocks stored more than once (same hash re-stored = actually reused)."""
        c = Counter()
        for e in self.events:
            if e.kind == "stored":
                for h in e.block_hashes:
                    c[h] += 1
        return sum(1 for h, n in c.items() if n > 1)

    def to_dict(self) -> dict:
        return {
            "num_events": len(self.events),
            "num_stored": self.num_stored,
            "num_removed": self.num_removed,
            "num_cleared": self.num_cleared,
            "total_stored_tokens": self.total_stored_tokens,
            "total_removed_tokens": self.total_removed_tokens,
            "churn_rate": round(self.churn_rate, 4),
            "blocks_stored": len(self.blocks_stored),
            "blocks_removed": len(self.blocks_removed),
            "reuse_confirmed": self.reuse_confirmed,
            "chained_stores": self.chained_stores,
            "medium_stored_tokens": dict(self.medium_stored_tokens),
            "medium_removed_tokens": dict(self.medium_removed_tokens),
        }

    def render_text(self) -> str:
        lines = [
            "KV Cache Event Stream Report",
            "=" * 40,
            f"events                : {len(self.events)}",
            f"stored / removed / clr: {self.num_stored} / {self.num_removed} / {self.num_cleared}",
            f"stored tokens         : {self.total_stored_tokens}",
            f"removed tokens        : {self.total_removed_tokens}",
            f"churn rate            : {self.churn_rate:.2%}",
            f"blocks stored/removed : {len(self.blocks_stored)} / {len(self.blocks_removed)}",
            f"reuse confirmed       : {self.reuse_confirmed} blocks re-stored",
            f"chained stores        : {self.chained_stores} (parent-linked)",
            f"medium movement       : stored {dict(self.medium_stored_tokens)} / removed {dict(self.medium_removed_tokens)}",
        ]
        if self.removal_timeline:
            lines.append("")
            lines.append("eviction timeline (first 10):")
            for seq, h, medium in self.removal_timeline[:10]:
                lines.append(f"  #{seq}: block {h} removed from {medium}")
            if len(self.removal_timeline) > 10:
                lines.append(f"  ... {len(self.removal_timeline) - 10} more")
        return "\n".join(lines)


def load_events(path: str | Path) -> EventStream:
    """Parse an events JSONL file into a metric-bearing stream."""
    records: list[EventRecord] = []
    with open(path, encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            kind = obj.get("type")
            if kind not in ("stored", "removed", "cleared"):
                raise ValueError(f"{path}:{line_no + 1}: unknown event type {kind!r}")
            rec = EventRecord(
                kind=kind,
                block_hashes=[int(h) for h in obj.get("block_hashes", [])],
                parent_block_hash=(
                    int(obj["parent_block_hash"])
                    if obj.get("parent_block_hash") is not None
                    else None
                ),
                num_tokens=int(obj.get("num_tokens", 0)),
                medium=str(obj.get("medium", "GPU")),
            )
            records.append(rec)

    # Assign sequence numbers and derive metrics.
    stream = EventStream(events=records)
    for i, rec in enumerate(records):
        rec.seq = i
        if rec.kind == "stored":
            stream.num_stored += 1
            stream.total_stored_tokens += rec.num_tokens
            stream.medium_stored_tokens[rec.medium] = (
                stream.medium_stored_tokens.get(rec.medium, 0) + rec.num_tokens
            )
            for h in rec.block_hashes:
                stream.blocks_stored.add(h)
            if rec.parent_block_hash is not None:
                stream.chained_stores += 1
        elif rec.kind == "removed":
            stream.num_removed += 1
            stream.total_removed_tokens += len(rec.block_hashes)  # tokens unknown; count blocks
            stream.medium_removed_tokens[rec.medium] = (
                stream.medium_removed_tokens.get(rec.medium, 0) + len(rec.block_hashes)
            )
            for h in rec.block_hashes:
                stream.blocks_removed.add(h)
                stream.removal_timeline.append((i, h, rec.medium))
        else:  # cleared
            stream.num_cleared += 1

    return stream
