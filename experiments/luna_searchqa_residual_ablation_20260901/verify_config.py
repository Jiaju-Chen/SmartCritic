#!/usr/bin/env python3
"""Reject configuration drift beyond the direct-vs-residual ablation."""

import argparse
import json
import re
from pathlib import Path

from omegaconf import OmegaConf


ALLOWED = {
    "algorithm.hybrid_advantage.composition_mode",
    "trainer.experiment_name",
    "trainer.default_local_dir",
    "ray_init._temp_dir",
}


def flatten(value, prefix=""):
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            result.update(flatten(child, f"{prefix}.{key}" if prefix else key))
        return result
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+(?:\.\d+)?[eE][+-]?\d+", value):
        value = float(value)
    return {prefix: value}


def compare(candidate, reference):
    candidate = OmegaConf.to_container(OmegaConf.create(candidate), resolve=True)
    for section in [candidate["actor_rollout_ref"]["actor"], candidate["critic"]]:
        if section["optim"]["total_training_steps"] == -1:
            section["optim"]["total_training_steps"] = candidate["trainer"]["total_training_steps"]

    old, new = flatten(reference), flatten(candidate)
    differences = {
        key: {"reference": old.get(key), "candidate": new.get(key)}
        for key in sorted(old.keys() | new.keys())
        if key not in old or key not in new or old[key] != new[key]
    }
    unexpected = sorted(set(differences) - ALLOWED)

    if old["algorithm.hybrid_advantage.composition_mode"] != "residual":
        raise ValueError("Reference composition must be residual")
    if new["algorithm.hybrid_advantage.composition_mode"] != "direct":
        raise ValueError("Candidate composition must be direct")
    if old["algorithm.hybrid_advantage.whiten_advantages"] is not False:
        raise ValueError("Reference final mixed advantages must not be whitened")
    if new["algorithm.hybrid_advantage.whiten_advantages"] is not False:
        raise ValueError("Candidate final mixed advantages must not be whitened")

    return {
        "passed": not unexpected,
        "differences": differences,
        "unexpected_keys": unexpected,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    reference_path = (
        Path(__file__).parents[1]
        / "luna_searchqa_whitening_ablation_20260830"
        / "reference_config.json"
    )
    reference = json.loads(reference_path.read_text())
    candidate = OmegaConf.to_container(OmegaConf.load(args.candidate), resolve=True)
    report = compare(candidate, reference)
    print(json.dumps(report, indent=2, ensure_ascii=True))
    if not report["passed"]:
        raise SystemExit("Unexpected configuration differences; refusing to launch")


if __name__ == "__main__":
    main()
