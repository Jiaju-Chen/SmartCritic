#!/usr/bin/env python3
"""Create a deterministic, source-stratified SearchQA validation subset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def proportional_quotas(counts: pd.Series, total: int) -> dict[str, int]:
    """Allocate exactly ``total`` rows using the largest-remainder rule."""
    raw = counts / counts.sum() * total
    quotas = raw.astype(int)
    remainder = raw - quotas
    missing = total - int(quotas.sum())
    for source in remainder.sort_values(ascending=False).index[:missing]:
        quotas.loc[source] += 1
    return {str(source): int(quota) for source, quota in quotas.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.size <= 0:
        raise ValueError("size must be positive")

    data = pd.read_parquet(args.input)
    if len(data) < args.size:
        raise ValueError(f"requested {args.size} rows, but input has {len(data)}")
    if "data_source" not in data.columns:
        raise KeyError("SearchQA data must contain data_source")

    quotas = proportional_quotas(data["data_source"].value_counts(), args.size)
    selected_parts = []
    for source, quota in quotas.items():
        source_rows = data[data["data_source"] == source]
        selected_parts.append(source_rows.sample(n=quota, random_state=args.seed))

    selected = pd.concat(selected_parts, axis=0).sort_index()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(args.output, index=False)

    print(f"Wrote {len(selected)} rows to {args.output}")
    print("Source counts:")
    print(selected["data_source"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
