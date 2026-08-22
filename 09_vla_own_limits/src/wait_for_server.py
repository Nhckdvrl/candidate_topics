"""Block until a Topic-09 policy server accepts connections (i.e. finished loading)."""
from __future__ import annotations

import argparse
import socket
import sys
import time


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--timeout", type=float, default=1800.0)
    args = p.parse_args()

    # The server binds only after create_trained_policy() returns, so an accepted
    # connection is a genuine readiness signal rather than a liveness one.
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        with socket.socket() as s:
            s.settimeout(2.0)
            if s.connect_ex((args.host, args.port)) == 0:
                print(f"server ready on {args.host}:{args.port}")
                return
        time.sleep(5.0)
    print(f"timed out waiting for {args.host}:{args.port}", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
