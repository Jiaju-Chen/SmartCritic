#!/usr/bin/env python3
import argparse
import concurrent.futures
import json
import statistics
import time

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18002/retrieve")
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()

    def request_one(index):
        start = time.perf_counter()
        response = requests.post(
            args.url,
            json={"query": QUERIES[index % len(QUERIES)], "topk": 3, "return_scores": False},
            timeout=args.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        result = payload.get("result")
        if not result:
            raise RuntimeError("Retriever returned an empty result payload")
        return time.perf_counter() - start

    request_one(0)
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        latencies = list(executor.map(request_one, range(args.requests)))
    elapsed = time.perf_counter() - started
    ordered = sorted(latencies)
    p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    print(
        json.dumps(
            {
                "requests": args.requests,
                "concurrency": args.concurrency,
                "elapsed_seconds": elapsed,
                "requests_per_second": args.requests / elapsed,
                "latency_mean_seconds": statistics.mean(latencies),
                "latency_p50_seconds": statistics.median(latencies),
                "latency_p95_seconds": ordered[p95_index],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
