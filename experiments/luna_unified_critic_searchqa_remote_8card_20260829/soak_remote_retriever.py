#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
import os
import statistics
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


FIXED_QUERY = "Who wrote Pride and Prejudice?"
EXPECTED_IDS = ["4950710", "592735", "13328669"]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def percentile(values, quantile):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * quantile))
    return ordered[index]


def retrieve(url, query, topk=3, timeout=30):
    payload = json.dumps(
        {"query": query, "topk": topk, "return_scores": False}
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    elapsed = time.perf_counter() - start
    documents = result["result"][0]
    if len(documents) != topk:
        raise RuntimeError(f"expected {topk} documents, received {len(documents)}")
    return elapsed, [str(document["id"]) for document in documents]


def run_burst(url, concurrency, timeout, burst_index):
    latencies = []
    errors = []
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                retrieve,
                url,
                f"SearchQA tunnel stability query {burst_index}-{index}",
                3,
                timeout,
            )
            for index in range(concurrency)
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                latency, _ = future.result()
                latencies.append(latency)
            except Exception as error:  # noqa: BLE001 - diagnostic must log all failures
                errors.append(repr(error))
    return {
        "kind": "burst",
        "timestamp": utc_now(),
        "concurrency": concurrency,
        "successes": len(latencies),
        "failures": len(errors),
        "wall_seconds": time.perf_counter() - started,
        "p50_seconds": percentile(latencies, 0.50),
        "p95_seconds": percentile(latencies, 0.95),
        "max_seconds": max(latencies) if latencies else None,
        "errors": errors[:5],
    }


def append_record(path, record):
    line = json.dumps(record, ensure_ascii=True, sort_keys=True)
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(line + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18003/retrieve")
    parser.add_argument("--duration-seconds", type=int, default=21600)
    parser.add_argument("--probe-interval-seconds", type=int, default=30)
    parser.add_argument("--burst-interval-seconds", type=int, default=300)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--startup-timeout-seconds", type=int, default=120)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for variable in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        os.environ.pop(variable, None)
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    os.environ["no_proxy"] = os.environ["NO_PROXY"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    probe_latencies = []
    probe_failures = 0
    consistency_failures = 0
    burst_failures = 0
    burst_index = 0

    append_record(
        args.output,
        {
            "kind": "start",
            "timestamp": utc_now(),
            "url": args.url,
            "duration_seconds": args.duration_seconds,
            "probe_interval_seconds": args.probe_interval_seconds,
            "burst_interval_seconds": args.burst_interval_seconds,
            "concurrency": args.concurrency,
        },
    )

    startup_deadline = time.monotonic() + args.startup_timeout_seconds
    while True:
        try:
            latency, ids = retrieve(
                args.url, FIXED_QUERY, 3, args.timeout_seconds
            )
            if ids != EXPECTED_IDS:
                raise RuntimeError(
                    f"startup result mismatch: expected {EXPECTED_IDS}, received {ids}"
                )
            append_record(
                args.output,
                {
                    "kind": "ready",
                    "timestamp": utc_now(),
                    "latency_seconds": latency,
                    "ids": ids,
                },
            )
            break
        except Exception as error:  # noqa: BLE001 - startup diagnostics are retained
            append_record(
                args.output,
                {
                    "kind": "startup_wait",
                    "timestamp": utc_now(),
                    "error": repr(error),
                },
            )
            if time.monotonic() >= startup_deadline:
                raise SystemExit("retriever did not become ready before startup timeout")
            time.sleep(2)

    started = time.monotonic()
    deadline = started + args.duration_seconds
    next_burst = started

    while time.monotonic() < deadline:
        try:
            latency, ids = retrieve(
                args.url, FIXED_QUERY, 3, args.timeout_seconds
            )
            consistent = ids == EXPECTED_IDS
            probe_latencies.append(latency)
            consistency_failures += int(not consistent)
            append_record(
                args.output,
                {
                    "kind": "probe",
                    "timestamp": utc_now(),
                    "latency_seconds": latency,
                    "ids": ids,
                    "consistent": consistent,
                },
            )
        except Exception as error:  # noqa: BLE001 - diagnostic must continue
            probe_failures += 1
            append_record(
                args.output,
                {
                    "kind": "probe",
                    "timestamp": utc_now(),
                    "error": repr(error),
                    "consistent": False,
                },
            )

        now = time.monotonic()
        if now >= next_burst:
            burst_index += 1
            record = run_burst(
                args.url, args.concurrency, args.timeout_seconds, burst_index
            )
            burst_failures += record["failures"]
            append_record(args.output, record)
            next_burst = now + args.burst_interval_seconds

        sleep_seconds = min(
            args.probe_interval_seconds, max(0, deadline - time.monotonic())
        )
        time.sleep(sleep_seconds)

    summary = {
        "kind": "summary",
        "timestamp": utc_now(),
        "duration_seconds": time.monotonic() - started,
        "probe_successes": len(probe_latencies),
        "probe_failures": probe_failures,
        "consistency_failures": consistency_failures,
        "burst_request_failures": burst_failures,
        "probe_mean_seconds": (
            statistics.fmean(probe_latencies) if probe_latencies else None
        ),
        "probe_p50_seconds": percentile(probe_latencies, 0.50),
        "probe_p95_seconds": percentile(probe_latencies, 0.95),
        "probe_max_seconds": max(probe_latencies) if probe_latencies else None,
        "passed": (
            probe_failures == 0
            and consistency_failures == 0
            and burst_failures == 0
        ),
    }
    append_record(args.output, summary)
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
