#!/usr/bin/env python3
import csv
import json
import os
import re
from pathlib import Path


AUDIT_ROOT = Path(
    os.environ.get(
        "AUDIT_ROOT",
        "/home/dataset-local/cjj/RL/runs/dual_critic_unseen_audit_20260814",
    )
)
RESULT_KEYS = (
    "val/text/test_score",
    "val/success_rate",
    "val/full_success_rate",
    "val/monitor32_success_rate",
    "val/evaluated_cases",
    "val/pick_and_place_success_rate",
    "val/pick_two_obj_and_place_success_rate",
    "val/pick_clean_then_place_in_recep_success_rate",
    "val/pick_cool_then_place_in_recep_success_rate",
    "val/look_at_obj_in_light_success_rate",
    "val/pick_heat_then_place_in_recep_success_rate",
)


def parse_run(run_dir: Path):
    match = re.match(
        r"(?P<method>dual2|dual28)_(?P<slot>best|latest)_step"
        r"(?P<step>\d+)_unseen(?P<val_size>\d+)_seed(?P<seed>\d+)_.*",
        run_dir.name,
    )
    if not match:
        return None

    log_path = run_dir / "logs" / "eval.log"
    status_path = run_dir / "status.txt"
    if not log_path.exists() or not status_path.exists():
        return None

    status_match = re.search(r"status=(\d+)", status_path.read_text())
    if not status_match:
        return None

    text = log_path.read_text(errors="replace")
    metric_lines = re.findall(r"step:\d+ - ([^\r\n]+)", text)
    metrics = {}
    if metric_lines:
        line = metric_lines[-1]
        for key in RESULT_KEYS:
            value = re.search(re.escape(key) + r":(-?\d+(?:\.\d+)?)", line)
            if value:
                metrics[key] = float(value.group(1))

    return {
        **match.groupdict(),
        "step": int(match.group("step")),
        "val_size": int(match.group("val_size")),
        "seed": int(match.group("seed")),
        "status": int(status_match.group(1)),
        **metrics,
        "run_dir": str(run_dir),
        "log_path": str(log_path),
    }


def main():
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = [
        row
        for run_dir in AUDIT_ROOT.iterdir()
        if run_dir.is_dir() and (row := parse_run(run_dir)) is not None
    ]
    order = {("dual2", "best"): 0, ("dual2", "latest"): 1,
             ("dual28", "best"): 2, ("dual28", "latest"): 3}
    rows.sort(key=lambda row: (order[(row["method"], row["slot"])], row["seed"]))

    json_path = AUDIT_ROOT / "results.json"
    csv_path = AUDIT_ROOT / "results.csv"
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")

    fieldnames = (
        "method",
        "slot",
        "step",
        "val_size",
        "seed",
        "status",
        *RESULT_KEYS,
        "run_dir",
        "log_path",
    )
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            row["method"],
            row["slot"],
            f"step={row['step']}",
            f"status={row['status']}",
            f"success={row.get('val/success_rate')}",
            f"cases={row.get('val/evaluated_cases')}",
        )
    print(f"csv={csv_path}")
    print(f"json={json_path}")


if __name__ == "__main__":
    main()
