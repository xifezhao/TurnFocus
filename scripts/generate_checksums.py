#!/usr/bin/env python3
"""Generate cryptographic SHA-256 checksum manifest (SHA256SUMS.txt) for IEEE/ACM Artifacts Available badge."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_path(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    skip_dirs = {".git", "__pycache__", ".pytest_cache", "turnfocus.egg-info", "build", "dist"}
    entries = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if rel.name == "SHA256SUMS.txt" or any(part in skip_dirs for part in rel.parts):
            continue
        entries.append(f"{sha256_path(p)}  {rel.as_posix()}")

    out_file = root / "SHA256SUMS.txt"
    out_file.write_text("\n".join(entries) + "\n", encoding="utf-8")
    print(f"Wrote {len(entries)} SHA-256 entries to {out_file}")


if __name__ == "__main__":
    main()
