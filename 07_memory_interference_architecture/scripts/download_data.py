from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/zhuangziGiantfish/Unable-to-Forget/main/testing_data/dict_category_double-word_46-400_v1-1.json"
UPSTREAM_GIT_BLOB = "15442a4cd50a7af5b9362620bbf43f6a0365965a"


def git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/dict_category_double-word_46-400_v1-1.json")
    args = p.parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=30) as r:
        data = r.read()
    actual_blob = git_blob_sha1(data)
    if actual_blob != UPSTREAM_GIT_BLOB:
        raise RuntimeError(
            f"upstream dataset changed: expected Git blob {UPSTREAM_GIT_BLOB}, got {actual_blob}"
        )
    out.write_bytes(data)
    print(f"wrote {out} ({len(data)} bytes)")
    print(f"sha256={hashlib.sha256(data).hexdigest()}")
    print(f"verified_upstream_git_blob={actual_blob}")


if __name__ == "__main__":
    main()
