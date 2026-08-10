#!/usr/bin/env python3
import csv
import json
import re
from pathlib import Path


AUDIT_ROOT = Path(
    "/home/dataset-local/cjj/RL/runs/full140_checkpoint_audit_20260810"
)
RESULT_KEYS = (
    "val/text/test_score",
    "val/success_rate",
    "val/full_success_rate",
    "val/monitor32_success_rate",
    "val/evaluated_cases",
)


def parse_run(run_dir: Path):
    match = re.match(
        r"(?P<method>critic2l|sao_skipobs)_step(?P<step>\d+)_seen"
        r"(?P<val_size>\d+)_seed(?P<seed>\d+)_.*",
        run_dir.name,
    )
    if not match:
        return None

    log_path = run_dir / "logs" / "eval.log"
    status_path = run_dir / "status.txt"
    if not log_path.exists():
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

    status = None
    if status_path.exists():
        value = re.search(r"status=(\d+)", status_path.read_text())
        if value:
            status = int(value.group(1))

    return {
        **match.groupdict(),
        "step": int(match.group("step")),
        "val_size": int(match.group("val_size")),
        "seed": int(match.group("seed")),
        "status": status,
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
    rows.sort(key=lambda row: (row["method"], row["step"], row["seed"]))

    json_path = AUDIT_ROOT / "results.json"
    csv_path = AUDIT_ROOT / "results.csv"
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")

    fieldnames = (
        "method",
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
            row["step"],
            f"status={row['status']}",
            f"success={row.get('val/success_rate')}",
            f"cases={row.get('val/evaluated_cases')}",
        )
    print(f"csv={csv_path}")
    print(f"json={json_path}")


if __name__ == "__main__":
    main()

