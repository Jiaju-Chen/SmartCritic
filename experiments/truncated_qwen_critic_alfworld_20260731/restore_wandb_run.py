#!/usr/bin/env python3
"""Restore numeric training history from the console log into a W&B run."""

from __future__ import annotations

import argparse
import math
import netrc
import os
import re
from pathlib import Path


STEP_RE = re.compile(r"step:(\d+) - (.*)")


def parse_history(log_path: Path) -> dict[int, dict[str, float]]:
    """Return the last complete numeric metric record for every training step."""
    history: dict[int, dict[str, float]] = {}
    for line in log_path.read_text(errors="replace").splitlines():
        match = STEP_RE.search(line)
        if match is None:
            continue

        step = int(match.group(1))
        metrics: dict[str, float] = {}
        for item in match.group(2).split(" - "):
            key, separator, value = item.rpartition(":")
            if not separator:
                continue
            try:
                number = float(value)
            except ValueError:
                continue
            if math.isfinite(number):
                metrics[key] = number

        if metrics:
            metrics["training/global_step"] = float(step)
            history[step] = metrics

    return history


def validate_history(history: dict[int, dict[str, float]], expected_steps: int) -> None:
    expected = set(range(1, expected_steps + 1))
    actual = set(history)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(f"invalid step coverage: missing={missing}, extra={extra}")

    validation_steps = [step for step, metrics in history.items() if "val/success_rate" in metrics]
    expected_validation_steps = list(range(5, expected_steps + 1, 5))
    if validation_steps != expected_validation_steps:
        raise ValueError(
            "invalid validation coverage: "
            f"expected={expected_validation_steps}, actual={validation_steps}"
        )


def load_api_key() -> str:
    auth = netrc.netrc().authenticators("api.wandb.ai")
    if auth is None or not auth[2]:
        raise RuntimeError("No api.wandb.ai credential found in the current user's .netrc")
    return auth[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--entity", default="cjj01-ustc")
    parser.add_argument("--project", default="verl_agent_alfworld_critic_ablation")
    parser.add_argument("--run-id", default="c2lppo0731-cjj-restored")
    parser.add_argument(
        "--name",
        default="ppo_qwen25_15b_critic2l_t128_v32_8gpu_seed0_20260731_restored",
    )
    parser.add_argument("--expected-steps", type=int, default=150)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    history = parse_history(args.log)
    validate_history(history, args.expected_steps)
    best_step = max(history, key=lambda step: history[step].get("val/success_rate", -math.inf))
    validation_count = sum("val/success_rate" in metrics for metrics in history.values())
    print(
        f"validated steps=1..{args.expected_steps}, validation_points={validation_count}, "
        f"best_val_step={best_step}, "
        f"best_val_success_rate={history[best_step]['val/success_rate']:.6f}",
        flush=True,
    )
    if args.validate_only:
        return

    os.environ["WANDB_API_KEY"] = load_api_key()
    os.environ["WANDB_ENTITY"] = args.entity
    os.environ["WANDB_PROJECT"] = args.project
    os.environ["WANDB_RUN_ID"] = args.run_id
    os.environ["WANDB_RESUME"] = "allow"

    import wandb

    run = wandb.init(
        entity=args.entity,
        project=args.project,
        id=args.run_id,
        name=args.name,
        resume="allow",
        tags=["restored", "ppo", "alfworld", "two-layer-critic"],
        notes=(
            "Restored from the completed local console log. Actor is full "
            "Qwen2.5-1.5B-Instruct; critic uses the embedding, first two Qwen layers, "
            "and a scalar value head."
        ),
        config={
            "restored_from_local_log": True,
            "source_run_id": "c2lppo0731",
            "actor_model": "Qwen/Qwen2.5-1.5B-Instruct",
            "critic_num_hidden_layers": 2,
            "critic_parameter_count": 326_972_417,
            "train_data_size": 128,
            "val_data_size": 32,
            "env_rollout_n": 1,
            "total_epochs": 150,
            "test_freq": 5,
            "save_freq": 25,
            "max_env_steps": 50,
            "history_length": 2,
            "seed": 0,
            "actor_learning_rate": 1e-6,
            "critic_learning_rate": 1e-5,
            "ppo_mini_batch_size": 256,
            "num_gpus": 8,
        },
    )
    if run is None:
        raise RuntimeError("wandb.init() returned no run")

    run.define_metric("training/global_step")
    run.define_metric("*", step_metric="training/global_step")
    for step in range(1, args.expected_steps + 1):
        run.log(history[step], step=step, commit=True)

    run.summary["restoration/complete"] = True
    run.summary["restoration/source_steps"] = args.expected_steps
    run.summary["restoration/validation_points"] = validation_count
    run.summary["restoration/best_val_step"] = best_step
    run.summary["restoration/best_val_success_rate"] = history[best_step]["val/success_rate"]
    run.finish(exit_code=0)
    print(f"restored_run_url=https://wandb.ai/{args.entity}/{args.project}/runs/{args.run_id}")


if __name__ == "__main__":
    main()
