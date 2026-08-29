#!/usr/bin/env python3
"""Fail-fast HTTP acceptance test for the local official SearchQA retriever."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
from typing import Any

import requests


QUERIES = [
    "Who wrote Pride and Prejudice?",
    "What is the capital of Australia?",
    "Where was Alan Turing born?",
    "Who discovered penicillin?",
    "When was the Eiffel Tower completed?",
    "What river flows through Budapest?",
    "Who directed Spirited Away?",
    "What element has atomic number 79?",
]


def request_one(url: str, index: int, timeout: float) -> float:
    started = time.perf_counter()
    response = requests.post(
        url,
        json={"query": QUERIES[index % len(QUERIES)], "topk": 3, "return_scores": False},
        timeout=timeout,
    )
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    results = payload.get("result")
    if not isinstance(results, list) or not results or not results[0]:
        raise RuntimeError(f"empty retriever result: {payload!r}")
    return time.perf_counter() - started


def run_case(url: str, requests_count: int, concurrency: int, timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    failures: list[str] = []
    latencies: list[float] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(request_one, url, i, timeout) for i in range(requests_count)]
        for future in futures:
            try:
                latencies.append(future.result())
            except Exception as exc:  # noqa: BLE001 - report every failed request
                failures.append(f"{type(exc).__name__}: {exc}")

    elapsed = time.perf_counter() - started
    ordered = sorted(latencies)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))] if ordered else None
    return {
        "requests": requests_count,
        "concurrency": concurrency,
        "successful": len(latencies),
        "failed": len(failures),
        "elapsed_seconds": elapsed,
        "requests_per_second": requests_count / elapsed,
        "latency_mean_seconds": statistics.mean(latencies) if latencies else None,
        "latency_p50_seconds": statistics.median(latencies) if latencies else None,
        "latency_p95_seconds": p95,
        "failures": failures[:5],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18002/retrieve")
    parser.add_argument("--health-url", default="http://127.0.0.1:18002/health")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--sequential-requests", type=int, default=8)
    parser.add_argument("--parallel-requests", type=int, default=32)
    parser.add_argument("--parallel-concurrency", type=int, default=8)
    args = parser.parse_args()

    health = requests.get(args.health_url, timeout=5)
    health.raise_for_status()
    print(json.dumps({"health": health.json()}, indent=2))

    warmup = request_one(args.url, 0, args.timeout)
    print(json.dumps({"warmup_latency_seconds": warmup}, indent=2))

    cases = [
        run_case(args.url, args.sequential_requests, 1, args.timeout),
        run_case(args.url, args.parallel_requests, args.parallel_concurrency, args.timeout),
    ]
    print(json.dumps({"cases": cases}, indent=2))
    if any(case["failed"] for case in cases):
        raise SystemExit("retriever acceptance failed: at least one request failed")


if __name__ == "__main__":
    main()
