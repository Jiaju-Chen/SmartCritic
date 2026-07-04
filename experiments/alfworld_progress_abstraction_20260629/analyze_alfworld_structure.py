#!/usr/bin/env python3
"""Offline diagnostics for ALFWorld progress-abstraction research.

This script intentionally avoids importing the training stack.  It reads:
  1. ALFWorld expert trajectory JSON files.
  2. Optional eval logs from GraphGPO/GRPO/PPO runs.

The goal is to quantify whether raw observation/state grouping is too sparse
and whether task-progress abstractions create denser, more meaningful groups.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


TASK_PREFIXES = (
    "pick_and_place_simple",
    "pick_and_place_with_movable_recep",
    "pick_two_obj_and_place",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
)


ACTION_RE = re.compile(r"^(?P<verb>[a-z_]+)(?: (?P<rest>.*))?$")
PIPE_RE = re.compile(r"^(?P<verb>[a-z_]+)\|(?P<rest>.*)$")
TAKE_RE = re.compile(r"^take (?P<object>.+?) from (?P<source>.+)$")
PUT_RE = re.compile(r"^put (?P<object>.+?) in/on (?P<target>.+)$")
GO_RE = re.compile(r"^go to (?P<target>.+)$")
OPEN_RE = re.compile(r"^open (?P<target>.+)$")
CLOSE_RE = re.compile(r"^close (?P<target>.+)$")
HEAT_RE = re.compile(r"^heat (?P<object>.+?) with (?P<tool>.+)$")
COOL_RE = re.compile(r"^cool (?P<object>.+?) with (?P<tool>.+)$")
CLEAN_RE = re.compile(r"^clean (?P<object>.+?) with (?P<tool>.+)$")
USE_RE = re.compile(r"^use (?P<object>.+?) on (?P<target>.+)$")


@dataclass
class TrajectoryRecord:
    split: str
    task_type: str
    task_dir: str
    traj_path: str
    goal: str
    expert_actions: list[str]
    low_actions: list[str]


def iter_traj_paths(root: Path, splits: Iterable[str]) -> Iterable[Path]:
    for split in splits:
        split_dir = root / "json_2.1.1" / split
        if not split_dir.exists():
            continue
        yield from split_dir.glob("*/*/traj_data.json")


def task_type_from_dir(path: Path) -> str:
    task_dir = path.parent.parent.name
    for prefix in TASK_PREFIXES:
        if task_dir.startswith(prefix + "-"):
            return prefix
    return task_dir.split("-")[0]


def load_goal(data: dict[str, Any]) -> str:
    turk = data.get("turk_annotations", {})
    anns = turk.get("anns", [])
    if anns and isinstance(anns[0], dict):
        task_desc = anns[0].get("task_desc")
        if task_desc:
            return str(task_desc)
    return ""


def load_expert_actions(data: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    plan = data.get("plan", {})
    high = plan.get("high_pddl", [])
    for item in high:
        discrete = item.get("discrete_action", {}) if isinstance(item, dict) else {}
        action = discrete.get("action")
        args = discrete.get("args", [])
        if action:
            joined = " ".join([str(action).lower()] + [str(x).lower() for x in args])
            actions.append(canonicalize_action(joined))
    return actions


def load_low_actions(data: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    plan = data.get("plan", {})
    low = plan.get("low_actions", [])
    for item in low:
        if not isinstance(item, dict):
            continue
        api = item.get("api_action", {})
        action = api.get("action")
        if action:
            actions.append(str(action).lower())
    return actions


def canonicalize_object_name(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\b\d+\b", "<id>", text)
    text = text.replace("sliced", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def action_operator(action: str) -> tuple[str, str, str]:
    action = action.strip().lower()
    pipe = PIPE_RE.match(action)
    if pipe:
        verb = pipe.group("verb")
        rest = pipe.group("rest").strip()
        parts = [canonicalize_object_name(x) for x in rest.split() if x.strip()]
        if verb in {"gotolocation", "go"}:
            return "go", parts[0] if parts else "", ""
        if verb in {"pickupobject", "take"}:
            return "take", parts[0] if parts else "", ""
        if verb in {"putobject", "put"}:
            obj = parts[0] if parts else ""
            target = parts[-1] if len(parts) > 1 else ""
            return "put", obj, target
        if verb in {"openobject", "open"}:
            return "open", parts[0] if parts else "", ""
        if verb in {"closeobject", "close"}:
            return "close", parts[0] if parts else "", ""
        if verb in {"sliceobject", "slice"}:
            return "slice", parts[0] if parts else "", ""
        if verb in {"coolobject", "cool"}:
            return "cool", parts[0] if parts else "", ""
        if verb in {"heatobject", "heat"}:
            return "heat", parts[0] if parts else "", ""
        if verb in {"cleanobject", "clean"}:
            return "clean", parts[0] if parts else "", ""
        if verb in {"toggleobject", "toggle"}:
            return "toggle", parts[0] if parts else "", ""
        if verb == "noop":
            return "noop", "", ""
        return verb, " ".join(parts), ""

    for regex, op in (
        (TAKE_RE, "take"),
        (PUT_RE, "put"),
        (GO_RE, "go"),
        (OPEN_RE, "open"),
        (CLOSE_RE, "close"),
        (HEAT_RE, "heat"),
        (COOL_RE, "cool"),
        (CLEAN_RE, "clean"),
        (USE_RE, "use"),
    ):
        match = regex.match(action)
        if match:
            groups = match.groupdict()
            obj = canonicalize_object_name(groups.get("object", ""))
            target = canonicalize_object_name(groups.get("target") or groups.get("source") or groups.get("tool") or "")
            return op, obj, target
    if action in {"look", "inventory"}:
        return action, "", ""
    match = ACTION_RE.match(action)
    if match:
        return match.group("verb"), canonicalize_object_name(match.group("rest") or ""), ""
    return "unknown", action, ""


def canonicalize_action(action: str) -> str:
    op, obj, target = action_operator(action)
    parts = [op]
    if obj:
        parts.append(obj)
    if target:
        parts.append(target)
    return "|".join(parts)


def progress_state_after_actions(task_type: str, goal: str, actions: list[str], upto: int) -> tuple[Any, ...]:
    """A deliberately coarse task-progress abstraction.

    It does not use simulator internals.  It only tracks progress-relevant event
    types and target classes inferred from the action strings.
    """
    prefix = task_type
    holding = "none"
    picked = 0
    placed = 0
    opened = 0
    searched = 0
    transformed = set()
    visited_targets = set()
    last_nav = "start"

    for action in actions[:upto]:
        op, obj, target = action_operator(action)
        if op == "go":
            last_nav = target or obj
            visited_targets.add(last_nav)
        elif op == "open":
            opened += 1
            searched += 1
            visited_targets.add(obj)
        elif op in {"look", "examine"}:
            searched += 1
        elif op == "take":
            picked += 1
            holding = obj or "object"
        elif op == "put":
            placed += 1
            holding = "none"
            visited_targets.add(target)
        elif op in {"heat", "cool", "clean", "use", "slice", "toggle"}:
            transformed.add(op)

    required_transform = "none"
    if "heat" in prefix:
        required_transform = "heat"
    elif "cool" in prefix:
        required_transform = "cool"
    elif "clean" in prefix:
        required_transform = "clean"
    elif "look_at" in prefix:
        required_transform = "look"

    if "two_obj" in prefix:
        pick_bucket = min(picked, 2)
        place_bucket = min(placed, 2)
    else:
        pick_bucket = min(picked, 1)
        place_bucket = min(placed, 1)

    transform_done = required_transform in transformed or required_transform == "none"
    if required_transform == "look":
        transform_done = "toggle" in transformed or searched > 0

    return (
        prefix,
        pick_bucket,
        place_bucket,
        holding != "none",
        transform_done,
        min(opened, 3),
        min(searched, 5),
        min(len(visited_targets), 10),
        canonicalize_object_name(last_nav),
    )


def progress_state_after_actions_medium(task_type: str, goal: str, actions: list[str], upto: int) -> tuple[Any, ...]:
    """A stricter abstraction that preserves object/receptacle classes.

    This is meant to sit between raw history and the very coarse progress
    phase abstraction. It keeps task-relevant object and receptacle classes but
    drops instance ids and exact navigation order.
    """
    coarse = progress_state_after_actions(task_type, goal, actions, upto)
    seen_locations = set()
    touched_objects = set()
    target_receptacles = set()
    transform_ops = set()
    last_op = "start"
    for action in actions[:upto]:
        op, obj, target = action_operator(action)
        last_op = op
        if op == "go" and obj:
            seen_locations.add(obj)
        if obj and op in {"take", "put", "heat", "cool", "clean", "slice", "toggle"}:
            touched_objects.add(obj)
        if target and op == "put":
            target_receptacles.add(target)
        if op in {"heat", "cool", "clean", "slice", "toggle"}:
            transform_ops.add(op)
    return (
        coarse[:8],
        tuple(sorted(list(touched_objects))[:3]),
        tuple(sorted(list(target_receptacles))[:2]),
        tuple(sorted(transform_ops)),
        min(len(seen_locations), 12),
        last_op,
    )


def parse_eval_log(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    text = path.read_text(errors="replace")
    final_status = None
    status_match = re.search(r"finished status=(\d+)", text)
    if status_match:
        final_status = int(status_match.group(1))
    metric_lines = re.findall(r"step:(\d+) - ([^\n]+)", text)
    metrics: list[dict[str, Any]] = []
    for step, body in metric_lines:
        item: dict[str, Any] = {"step": int(step)}
        for part in body.split(" - "):
            if ":" not in part:
                continue
            key, val = part.split(":", 1)
            try:
                item[key.strip()] = float(val)
            except ValueError:
                item[key.strip()] = val.strip()
        metrics.append(item)
    return {
        "path": str(path),
        "exists": True,
        "bytes": path.stat().st_size,
        "finished_status": final_status,
        "last_metrics": metrics[-1] if metrics else None,
        "num_metric_lines": len(metrics),
    }


def load_records(root: Path, splits: list[str], limit: int | None) -> list[TrajectoryRecord]:
    records: list[TrajectoryRecord] = []
    for traj_path in iter_traj_paths(root, splits):
        with traj_path.open() as f:
            data = json.load(f)
        records.append(
            TrajectoryRecord(
                split=traj_path.parts[-4],
                task_type=task_type_from_dir(traj_path),
                task_dir=traj_path.parent.parent.name,
                traj_path=str(traj_path),
                goal=load_goal(data),
                expert_actions=load_expert_actions(data),
                low_actions=load_low_actions(data),
            )
        )
        if limit is not None and len(records) >= limit:
            break
    return records


def summarize_lengths(records: list[TrajectoryRecord]) -> dict[str, Any]:
    by_task: dict[str, list[int]] = collections.defaultdict(list)
    for r in records:
        by_task[r.task_type].append(len(r.expert_actions))
    summary = {}
    for task, vals in sorted(by_task.items()):
        summary[task] = {
            "n": len(vals),
            "mean_high_level_steps": mean(vals) if vals else 0.0,
            "median_high_level_steps": median(vals) if vals else 0.0,
            "p90_high_level_steps": percentile(vals, 90),
        }
    all_vals = [len(r.expert_actions) for r in records]
    summary["_overall"] = {
        "n": len(all_vals),
        "mean_high_level_steps": mean(all_vals) if all_vals else 0.0,
        "median_high_level_steps": median(all_vals) if all_vals else 0.0,
        "p90_high_level_steps": percentile(all_vals, 90),
    }
    return summary


def percentile(vals: list[int], p: float) -> float:
    if not vals:
        return 0.0
    vals = sorted(vals)
    idx = (len(vals) - 1) * p / 100.0
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return float(vals[lo])
    return float(vals[lo] * (hi - idx) + vals[hi] * (idx - lo))


def group_stats(keys: list[Any]) -> dict[str, Any]:
    counts = collections.Counter(keys)
    total = len(keys)
    repeated_items = sum(v for v in counts.values() if v > 1)
    repeated_groups = sum(1 for v in counts.values() if v > 1)
    return {
        "items": total,
        "unique_groups": len(counts),
        "compression": total / len(counts) if counts else 0.0,
        "repeated_item_fraction": repeated_items / total if total else 0.0,
        "repeated_groups": repeated_groups,
        "mean_group_size": total / len(counts) if counts else 0.0,
        "max_group_size": max(counts.values()) if counts else 0,
    }


def abstraction_density(records: list[TrajectoryRecord]) -> dict[str, Any]:
    raw_prefix_keys = []
    action_op_keys = []
    progress_coarse_keys = []
    progress_medium_keys = []
    progress_action_schema_keys = []
    progress_action_semantic_keys = []
    within_task_raw = collections.defaultdict(list)
    within_task_progress_coarse = collections.defaultdict(list)
    within_task_progress_medium = collections.defaultdict(list)

    for r in records:
        actions = r.expert_actions
        for step in range(len(actions) + 1):
            raw_prefix = tuple(actions[:step])
            progress_coarse = progress_state_after_actions(r.task_type, r.goal, actions, step)
            progress_medium = progress_state_after_actions_medium(r.task_type, r.goal, actions, step)
            raw_prefix_keys.append((r.task_type, raw_prefix))
            progress_coarse_keys.append(progress_coarse)
            progress_medium_keys.append(progress_medium)
            within_task_raw[r.task_type].append(raw_prefix)
            within_task_progress_coarse[r.task_type].append(progress_coarse)
            within_task_progress_medium[r.task_type].append(progress_medium)
            if step < len(actions):
                op, obj, target = action_operator(actions[step])
                action_op = (op, bool(obj), bool(target))
                action_op_keys.append(action_op)
                progress_action_schema_keys.append((progress_medium, action_op))
                progress_action_semantic_keys.append((progress_medium, op, obj, target))

    by_task = {}
    for task in sorted(within_task_raw):
        by_task[task] = {
            "raw_prefix": group_stats(within_task_raw[task]),
            "progress_coarse": group_stats(within_task_progress_coarse[task]),
            "progress_medium": group_stats(within_task_progress_medium[task]),
        }

    return {
        "raw_prefix_global": group_stats(raw_prefix_keys),
        "action_schema_global": group_stats(action_op_keys),
        "progress_coarse_global": group_stats(progress_coarse_keys),
        "progress_medium_global": group_stats(progress_medium_keys),
        "progress_medium_plus_action_schema": group_stats(progress_action_schema_keys),
        "progress_medium_plus_action_semantic": group_stats(progress_action_semantic_keys),
        "by_task": by_task,
    }


def action_distribution(records: list[TrajectoryRecord]) -> dict[str, Any]:
    op_counts: collections.Counter[str] = collections.Counter()
    task_op_counts: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for r in records:
        for action in r.expert_actions:
            op, _, _ = action_operator(action)
            op_counts[op] += 1
            task_op_counts[r.task_type][op] += 1
    return {
        "overall": dict(op_counts.most_common()),
        "by_task": {task: dict(counter.most_common()) for task, counter in sorted(task_op_counts.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alfworld-root", required=True)
    parser.add_argument("--splits", nargs="+", default=["valid_seen", "valid_unseen"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--eval-log", action="append", default=[])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.alfworld_root)
    records = load_records(root, args.splits, args.limit)
    result = {
        "alfworld_root": str(root),
        "splits": args.splits,
        "num_records": len(records),
        "lengths": summarize_lengths(records),
        "actions": action_distribution(records),
        "abstraction_density": abstraction_density(records),
        "eval_logs": [parse_eval_log(Path(p)) for p in args.eval_log],
        "sample_records": [asdict(r) for r in records[:5]],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps({
        "out": str(out),
        "num_records": len(records),
        "overall_length": result["lengths"]["_overall"],
        "raw_prefix": result["abstraction_density"]["raw_prefix_global"],
        "progress_coarse": result["abstraction_density"]["progress_coarse_global"],
        "progress_medium": result["abstraction_density"]["progress_medium_global"],
    }, indent=2))


if __name__ == "__main__":
    main()
