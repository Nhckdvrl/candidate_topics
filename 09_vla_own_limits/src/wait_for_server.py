"""Block until a Topic-09 policy server is ready to serve.

Uses the OpenPI websocket server's `/healthz` HTTP endpoint rather than a bare TCP
connect. A bare connect opens and drops a socket, which the websocket layer logs as
`InvalidMessage: did not receive a valid HTTP request` — noise in exactly the log a
person would read to diagnose a genuinely broken server.

The port only starts listening after `create_trained_policy` returns, so a successful
health check is a readiness signal, not merely a liveness one.
"""
from __future__ import annotations

import argparse
import http.client
import socket
import sys
import time


def _healthy(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
        try:
            conn.request("GET", "/healthz")
            return conn.getresponse().status == 200
        finally:
            conn.close()
    except (OSError, socket.timeout, http.client.HTTPException):
        return False


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--timeout", type=float, default=1800.0)
    p.add_argument("--poll", type=float, default=5.0)
    args = p.parse_args()

    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        if _healthy(args.host, args.port):
            print(f"server ready on {args.host}:{args.port}")
            return
        time.sleep(args.poll)
    print(f"timed out waiting for {args.host}:{args.port}", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
