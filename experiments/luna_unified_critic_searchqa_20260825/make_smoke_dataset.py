#!/usr/bin/env python3
"""Create a compact Search-R1-format dataset for an end-to-end smoke run."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CASES = [
    ("In which city is the Eiffel Tower located?", ["Paris"]),
    ("Who wrote Pride and Prejudice?", ["Jane Austen"]),
    ("What is the chemical symbol for gold?", ["Au"]),
    ("What is the largest planet in the Solar System?", ["Jupiter"]),
    ("In what year did Apollo 11 land on the Moon?", ["1969"]),
    ("What is the capital city of Australia?", ["Canberra"]),
    ("Who painted the Mona Lisa?", ["Leonardo da Vinci"]),
    ("At sea level, what is the boiling point of water in Celsius?", ["100 degrees Celsius", "100 C", "100"]),
]


def make_row(question: str, answers: list[str], split: str, index: int) -> dict:
    ground_truth = {"target": answers}
    data_source = "searchR1_smoke"
    prompt = [
        {"role": "system", "content": "You are a helpful and harmless assistant."},
        {"role": "user", "content": question},
    ]
    reward_model = {"ground_truth": ground_truth, "style": "rule"}
    return {
        "data_source": data_source,
        "prompt": prompt,
        "ability": "fact_retrieval",
        "reward_model": reward_model,
        "extra_info": {
            "index": index,
            "need_tools_kwargs": True,
            "question": question,
            "split": split,
            "tools_kwargs": {
                "search": {
                    "create_kwargs": {
                        "ground_truth": ground_truth,
                        "question": question,
                        "data_source": data_source,
                    }
                }
            },
        },
        "metadata": {"smoke": True},
        "env_kwargs": {
            "ground_truth": ground_truth,
            "question": question,
            "data_source": data_source,
        },
    }


def write_split(path: Path, split: str, size: int, offset: int) -> None:
    rows = []
    for row_index in range(size):
        question, answers = CASES[(row_index + offset) % len(CASES)]
        rows.append(make_row(question, answers, split, row_index))
    pd.DataFrame(rows).to_parquet(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=8)
    parser.add_argument("--val-size", type=int, default=4)
    args = parser.parse_args()

    if args.train_size <= 0 or args.val_size <= 0:
        raise ValueError("train-size and val-size must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_split(args.output_dir / "train.parquet", "train", args.train_size, 0)
    write_split(args.output_dir / "test.parquet", "test", args.val_size, 4)
    print(f"Wrote SearchQA smoke data to {args.output_dir}")


if __name__ == "__main__":
    main()
