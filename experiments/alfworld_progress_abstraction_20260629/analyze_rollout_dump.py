#!/usr/bin/env python3
"""Analyze step-level validation rollout dumps produced by GraphGPO."""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
from pathlib import Path
from statistics import mean, median
from typing import Any


TASK_RE = re.compile(r"Your task is to: (?P<goal>.+?)(?:\. Location:|$)", re.S)
LOCATION_RE = re.compile(r"Location: (?P<loc>.+?)\. Items in hand")
HOLDING_RE = re.compile(r"Items in hand \(status\): (?P<holding>.+?)\.")
ACTION_RE = re.compile(r"^(?P<verb>[a-z]+)(?: (?P<rest>.*))?$")
GO_RE = re.compile(r"^go to (?P<target>.+)$")
TAKE_RE = re.compile(r"^take (?P<object>.+?) from (?P<source>.+)$")
PUT_RE = re.compile(r"^put (?P<object>.+?) in/on (?P<target>.+)$")
MOVE_RE = re.compile(r"^move (?P<object>.+?) to (?P<target>.+)$")
OPEN_RE = re.compile(r"^open (?P<target>.+)$")
EXAMINE_RE = re.compile(r"^examine (?P<target>.+)$")
HEAT_RE = re.compile(r"^heat (?P<object>.+?) with (?P<tool>.+)$")
COOL_RE = re.compile(r"^cool (?P<object>.+?) with (?P<tool>.+)$")
CLEAN_RE = re.compile(r"^clean (?P<object>.+?) with (?P<tool>.+)$")


def norm(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"\b\d+\b", "<id>", text)
    text = text.replace("sliced", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_goal(obs: str) -> tuple[str, str, str]:
    match = TASK_RE.search(obs)
    goal = norm(match.group("goal")) if match else ""
    task_type = "unknown"
    if "find two" in goal:
        task_type = "pick_two_obj_and_place"
    elif "look at" in goal or "in light" in goal:
        task_type = "look_at_obj_in_light"
    elif "cool" in goal:
        task_type = "pick_cool_then_place_in_recep"
    elif "heat" in goal:
        task_type = "pick_heat_then_place_in_recep"
    elif "clean" in goal:
        task_type = "pick_clean_then_place_in_recep"
    elif "put" in goal:
        task_type = "pick_and_place"

    objects = re.findall(r"find(?: two)? (?P<object>[a-z]+(?: <id>)?)", goal)
    obj = objects[0] if objects else ""
    receptacle = ""
    m = re.search(r"put (?:them|it|.+?) in (?P<rec>[a-z]+(?: <id>)?)", goal)
    if m:
        receptacle = m.group("rec")
    return goal, task_type, obj, receptacle


def parse_obs(obs: str) -> dict[str, str]:
    loc = LOCATION_RE.search(obs)
    holding = HOLDING_RE.search(obs)
    goal, task_type, goal_obj, goal_rec = parse_goal(obs)
    return {
        "goal": goal,
        "task_type": task_type,
        "goal_obj": goal_obj,
        "goal_receptacle": goal_rec,
        "location": norm(loc.group("loc")) if loc else "",
        "holding": norm(holding.group("holding")) if holding else "",
    }


def action_operator(action: str) -> tuple[str, str, str]:
    action = norm(action)
    for regex, op in (
        (GO_RE, "go"),
        (TAKE_RE, "take"),
        (PUT_RE, "put"),
        (MOVE_RE, "move"),
        (OPEN_RE, "open"),
        (EXAMINE_RE, "examine"),
        (HEAT_RE, "heat"),
        (COOL_RE, "cool"),
        (CLEAN_RE, "clean"),
    ):
        m = regex.match(action)
        if m:
            g = m.groupdict()
            return op, norm(g.get("object") or g.get("target") or ""), norm(g.get("source") or g.get("tool") or "")
    if action in {"look", "inventory"}:
        return action, "", ""
    m = ACTION_RE.match(action)
    if m:
        return m.group("verb"), norm(m.group("rest") or ""), ""
    return "unknown", action, ""


def group_stats(keys: list[Any]) -> dict[str, Any]:
    counts = collections.Counter(keys)
    total = len(keys)
    repeated_items = sum(v for v in counts.values() if v > 1)
    return {
        "items": total,
        "unique_groups": len(counts),
        "compression": total / len(counts) if counts else 0.0,
        "repeated_item_fraction": repeated_items / total if total else 0.0,
        "mean_group_size": total / len(counts) if counts else 0.0,
        "max_group_size": max(counts.values()) if counts else 0,
    }


def progress_key(row: dict[str, Any], history: list[dict[str, Any]]) -> tuple[Any, ...]:
    obs = parse_obs(row.get("anchor_obs", ""))
    actions = [h["text_actions"] for h in history] + [row["text_actions"]]
    picked = placed = transformed = searched = 0
    visited = set()
    holding_goal = obs["goal_obj"] and obs["goal_obj"] in obs["holding"]
    current_loc = obs["location"]
    for action in actions:
        op, obj, target = action_operator(action)
        if op == "go":
            visited.add(obj)
        elif op in {"open", "examine", "look"}:
            searched += 1
        elif op == "take":
            if not obs["goal_obj"] or obs["goal_obj"] in obj:
                picked += 1
        elif op in {"put", "move"}:
            if not obs["goal_receptacle"] or obs["goal_receptacle"] in obj or obs["goal_receptacle"] in target:
                placed += 1
        elif op in {"heat", "cool", "clean"}:
            transformed += 1
    if "two_obj" in obs["task_type"]:
        pick_bucket = min(picked, 2)
        place_bucket = min(placed, 2)
    else:
        pick_bucket = min(picked, 1)
        place_bucket = min(placed, 1)
    return (
        obs["task_type"],
        obs["goal_obj"],
        obs["goal_receptacle"],
        current_loc,
        holding_goal,
        pick_bucket,
        place_bucket,
        min(transformed, 1),
        min(searched, 4),
        min(len(visited), 12),
    )


def load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open() as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    vals = sorted(vals)
    idx = (len(vals) - 1) * p / 100.0
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return float(vals[lo])
    return float(vals[lo] * (hi - idx) + vals[hi] * (idx - lo))


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows_by_traj: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        rows_by_traj[row["traj_uid"]].append(row)
    for traj_rows in rows_by_traj.values():
        traj_rows.sort(key=lambda x: x["row"])

    raw_obs_keys = []
    raw_obs_action_keys = []
    progress_keys = []
    progress_action_schema_keys = []
    progress_action_semantic_keys = []
    op_counts: collections.Counter[str] = collections.Counter()
    redundant_go = 0
    invalid = 0

    traj_summaries = []
    for traj_uid, traj_rows in rows_by_traj.items():
        seen_locs = set()
        history: list[dict[str, Any]] = []
        for row in traj_rows:
            obs = parse_obs(row.get("anchor_obs", ""))
            op, obj, target = action_operator(row.get("text_actions", ""))
            op_counts[op] += 1
            raw_obs_keys.append(norm(row.get("anchor_obs", "")))
            raw_obs_action_keys.append((norm(row.get("anchor_obs", "")), op, obj, target))
            pkey = progress_key(row, history)
            progress_keys.append(pkey)
            progress_action_schema_keys.append((pkey, op))
            progress_action_semantic_keys.append((pkey, op, obj, target))
            if op == "go":
                if obj in seen_locs:
                    redundant_go += 1
                seen_locs.add(obj)
            if not row.get("is_action_valid", True):
                invalid += 1
            history.append(row)

        first = traj_rows[0]
        obs0 = parse_obs(first.get("anchor_obs", ""))
        traj_summaries.append({
            "traj_uid": traj_uid,
            "task_type": obs0["task_type"],
            "goal": obs0["goal"],
            "steps": len(traj_rows),
            "episode_reward": float(first.get("episode_rewards", 0.0)),
            "success": float(first.get("episode_rewards", 0.0)) > 0,
            "unique_raw_obs": len({norm(r.get("anchor_obs", "")) for r in traj_rows}),
            "unique_progress": len({progress_key(r, traj_rows[:i]) for i, r in enumerate(traj_rows)}),
        })

    step_counts = [x["steps"] for x in traj_summaries]
    success_steps = [x["steps"] for x in traj_summaries if x["success"]]
    fail_steps = [x["steps"] for x in traj_summaries if not x["success"]]
    by_task = collections.defaultdict(list)
    for item in traj_summaries:
        by_task[item["task_type"]].append(item)

    return {
        "num_steps": len(rows),
        "num_trajectories": len(rows_by_traj),
        "success_rate": sum(1 for x in traj_summaries if x["success"]) / len(traj_summaries) if traj_summaries else 0.0,
        "steps": {
            "mean": mean(step_counts) if step_counts else 0.0,
            "median": median(step_counts) if step_counts else 0.0,
            "p90": percentile(step_counts, 90),
            "success_mean": mean(success_steps) if success_steps else 0.0,
            "fail_mean": mean(fail_steps) if fail_steps else 0.0,
        },
        "group_density": {
            "raw_obs": group_stats(raw_obs_keys),
            "raw_obs_plus_action": group_stats(raw_obs_action_keys),
            "progress": group_stats(progress_keys),
            "progress_plus_action_schema": group_stats(progress_action_schema_keys),
            "progress_plus_action_semantic": group_stats(progress_action_semantic_keys),
        },
        "action_ops": dict(op_counts.most_common()),
        "redundant_go_fraction": redundant_go / max(op_counts["go"], 1),
        "invalid_action_fraction": invalid / len(rows) if rows else 0.0,
        "by_task": {
            task: {
                "n": len(items),
                "success_rate": sum(1 for x in items if x["success"]) / len(items),
                "mean_steps": mean([x["steps"] for x in items]),
            }
            for task, items in sorted(by_task.items())
        },
        "trajectory_summaries": traj_summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = load_rows([Path(p) for p in args.dump])
    result = analyze(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps({
        "out": str(out),
        "num_steps": result["num_steps"],
        "num_trajectories": result["num_trajectories"],
        "success_rate": result["success_rate"],
        "group_density": result["group_density"],
        "steps": result["steps"],
    }, indent=2))


if __name__ == "__main__":
    main()
