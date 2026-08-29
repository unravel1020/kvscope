# SPDX-License-Identifier: Apache-2.0
"""KVScope command-line interface.

Usage mirrors vLLM's ``vllm analyze-prefix-cache`` subcommand shape:

    kvscope analyze --input snapshot.jsonl [--output-format text|json]
                   [--top-k-groups N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .events import load_events
from .report import build_report, render_text
from .snapshot import load_snapshot
from .tree import analyze_tree
from .turns import load_turns, summarize_turns


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kvscope",
        description="SGLang KV cache microscope — offline radix-cache analysis",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser(
        "analyze",
        help="analyze a radix-cache snapshot dump",
        description=(
            "Load a radix-cache snapshot JSONL and report structural, sharing, "
            "fragmentation, eviction and hotness metrics."
        ),
    )
    analyze.add_argument(
        "--input",
        type=Path,
        required=True,
        help="path to snapshot JSONL (see docs/DESIGN.md for schema)",
    )
    analyze.add_argument(
        "--output-format",
        choices=["text", "json"],
        default="text",
        help="report format (default: text)",
    )
    analyze.add_argument(
        "--top-k-groups",
        type=int,
        default=10,
        help="max hot nodes to report (default: 10)",
    )
    analyze.add_argument(
        "--small-node-threshold",
        type=int,
        default=16,
        help="token_len below which a node counts as fragmentation (default: 16)",
    )

    turns = sub.add_parser(
        "turns",
        help="analyze per-turn KV reuse of a multi-turn (agent) workload",
        description=(
            "Load a turns JSONL (per-turn hit_length/context_tokens) and "
            "report how KV reuse evolves across the conversation."
        ),
    )
    turns.add_argument(
        "--input",
        type=Path,
        required=True,
        help="path to turns JSONL (see scripts/simulate_agent_workload.py)",
    )
    turns.add_argument(
        "--output-format",
        choices=["text", "json"],
        default="text",
        help="report format (default: text)",
    )

    events = sub.add_parser(
        "events",
        help="analyze a KV cache event stream (placement/eviction timeline)",
        description=(
            "Load an events JSONL (BlockStored/BlockRemoved/AllBlocksCleared) "
            "and report cache lifecycle, churn and eviction timeline."
        ),
    )
    events.add_argument(
        "--input",
        type=Path,
        required=True,
        help="path to events JSONL (see scripts/simulate_agent_workload.py)",
    )
    events.add_argument(
        "--output-format",
        choices=["text", "json"],
        default="text",
        help="report format (default: text)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "analyze":
        try:
            snapshot = load_snapshot(args.input)
            metrics = analyze_tree(
                snapshot,
                small_node_threshold=args.small_node_threshold,
                top_hot=args.top_k_groups,
            )
        except (ValueError, OSError) as e:
            print(f"kvscope: error: {e}", file=sys.stderr)
            return 1

        report = build_report(snapshot, metrics)
        if args.output_format == "json":
            print(json.dumps(report, indent=2, ensure_ascii=False))
        else:
            print(render_text(report))
        return 0

    if args.command == "turns":
        try:
            records = load_turns(args.input)
            summary = summarize_turns(records)
        except (ValueError, OSError, json.JSONDecodeError) as e:
            print(f"kvscope: error: {e}", file=sys.stderr)
            return 1

        if args.output_format == "json":
            print(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False))
        else:
            print(summary.render_text())
        return 0

    if args.command == "events":
        try:
            stream = load_events(args.input)
        except (ValueError, OSError, json.JSONDecodeError) as e:
            print(f"kvscope: error: {e}", file=sys.stderr)
            return 1

        if args.output_format == "json":
            print(json.dumps(stream.to_dict(), indent=2, ensure_ascii=False))
        else:
            print(stream.render_text())
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
