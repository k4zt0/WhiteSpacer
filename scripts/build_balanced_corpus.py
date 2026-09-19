#!/usr/bin/env python3
"""Create a source-balanced split from the full provenance corpus."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        with path.open() as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        default=[
            Path("data/processed/train.jsonl"),
            Path("data/processed/validation.jsonl"),
        ],
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/balanced"))
    parser.add_argument("--nvd-limit", type=int, default=8_000)
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    parser.add_argument("--safety-repeat", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_rows(args.input):
        by_source[str(row["source"])].append(row)

    selected = []
    for source, rows in sorted(by_source.items()):
        rng.shuffle(rows)
        selected.extend(rows[: args.nvd_limit] if source == "NIST NVD" else rows)

    safety = [row for row in selected if row["source"] == "WhiteSpacer safety"]
    selected.extend(safety * (args.safety_repeat - 1))
    rng.shuffle(selected)
    validation_count = max(1, round(len(selected) * args.validation_ratio))
    validation = selected[:validation_count]
    train = selected[validation_count:]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "train.jsonl", train)
    write_rows(args.output_dir / "validation.jsonl", validation)
    manifest = {
        "total": len(selected),
        "train": len(train),
        "validation": len(validation),
        "train_sources": dict(sorted(Counter(row["source"] for row in train).items())),
        "validation_sources": dict(
            sorted(Counter(row["source"] for row in validation).items())
        ),
        "seed": args.seed,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()

