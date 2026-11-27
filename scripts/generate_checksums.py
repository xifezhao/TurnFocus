#!/usr/bin/env python3
"""Generate cryptographic SHA-256 checksum manifest (SHA256SUMS.txt) excluding OS and build metadata."""

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
    skip_dirs = {".git", "__pycache__", ".pytest_cache", "build", "dist"}
    skip_names = {"SHA256SUMS.txt", ".DS_Store"}
    entries = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if rel.name in skip_names or rel.name.startswith("._") or rel.name.endswith((".pyc", ".pyo", ".tar.gz")):
            continue
        if any(part in skip_dirs or part.endswith(".egg-info") for part in rel.parts):
            continue
        entries.append(f"{sha256_path(p)}  {rel.as_posix()}")

    out_file = root / "SHA256SUMS.txt"
    out_file.write_text("\n".join(entries) + "\n", encoding="utf-8")
    print(f"Wrote {len(entries)} clean SHA-256 entries to {out_file}")


if __name__ == "__main__":
    main()
