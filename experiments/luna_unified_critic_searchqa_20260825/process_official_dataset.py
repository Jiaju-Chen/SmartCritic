#!/usr/bin/env python3
"""Convert the official Search-R1 parquet files to verl-agent format."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SYSTEM_CONTENT = "You are a helpful and harmless assistant."


def process_row(row: pd.Series, split: str, index: int) -> pd.Series:
    question = row.get("question", "")
    reward_model = row.get("reward_model")
    if isinstance(reward_model, dict) and "ground_truth" in reward_model:
        ground_truth = reward_model["ground_truth"]
    else:
        ground_truth = row.get("golden_answers", [])

    data_source = str(row.get("data_source", ""))
    env_kwargs = {
        "ground_truth": ground_truth,
        "question": question,
        "data_source": data_source,
    }
    return pd.Series(
        {
            "data_source": data_source,
            "prompt": [
                {"role": "system", "content": SYSTEM_CONTENT},
                {"role": "user", "content": question},
            ],
            "ability": row.get("ability"),
            "reward_model": reward_model,
            "extra_info": {
                "index": index,
                "need_tools_kwargs": True,
                "question": question,
                "split": split,
                "tools_kwargs": {"search": {"create_kwargs": env_kwargs}},
            },
            "metadata": row.get("metadata"),
            "env_kwargs": env_kwargs,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        source = args.raw_dir / f"{split}.parquet"
        target = args.output_dir / f"{split}.parquet"
        frame = pd.read_parquet(source)
        processed = pd.DataFrame(
            [process_row(row, split, index) for index, (_, row) in enumerate(frame.iterrows())]
        )
        processed.to_parquet(target, index=False)
        counts = processed["data_source"].value_counts().sort_index().to_dict()
        print(f"{split}: rows={len(processed)} sources={counts} output={target}")


if __name__ == "__main__":
    main()
