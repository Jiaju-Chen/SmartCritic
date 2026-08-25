#!/usr/bin/env python3
"""Small CPU retriever implementing the Search-R1 `/retrieve` contract."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import re


DOCUMENTS = [
    ("Eiffel Tower", "The Eiffel Tower is a landmark located in Paris, France."),
    ("Pride and Prejudice", "Pride and Prejudice is a novel written by Jane Austen."),
    ("Gold", "Gold is a chemical element whose symbol is Au and atomic number is 79."),
    ("Jupiter", "Jupiter is the largest planet in the Solar System."),
    ("Apollo 11", "Apollo 11 landed on the Moon in 1969."),
    ("Canberra", "Canberra is the capital city of Australia."),
    ("Mona Lisa", "The Mona Lisa was painted by Leonardo da Vinci."),
    ("Boiling point", "At sea level, water boils at 100 degrees Celsius."),
]


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def retrieve(query: str, topk: int) -> list[dict]:
    query_tokens = tokens(query)
    ranked = []
    for title, body in DOCUMENTS:
        overlap = len(query_tokens & tokens(f"{title} {body}"))
        ranked.append((overlap, title, body))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    selected = ranked[: max(1, min(topk, len(ranked)))]
    return [
        {
            "document": {"contents": f"{title}\n{body}"},
            "score": float(score),
        }
        for score, title, body in selected
    ]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format_string: str, *args) -> None:
        print(f"{self.address_string()} - {format_string % args}", flush=True)

    def do_GET(self) -> None:
        if self.path != "/health":
            self.send_error(404)
            return
        self._send_json({"status": "ok"})

    def do_POST(self) -> None:
        if self.path != "/retrieve":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length) or b"{}")
        query = str(request.get("query", ""))
        topk = int(request.get("topk") or 3)
        results = retrieve(query, topk)
        print(f"query={query!r} results={len(results)}", flush=True)
        self._send_json({"result": [results]})

    def _send_json(self, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Mini Search-R1 retriever listening on {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
