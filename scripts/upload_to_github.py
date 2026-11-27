#!/usr/bin/env python3
"""Zero-dependency GitHub Git Data API uploader (pushes full artifact/ directory in one atomic commit).

Usage:
    export GITHUB_TOKEN="ghp_..."
    python3 scripts/upload_to_github.py --owner xifezhao --repo TurnFocus --branch main
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List


def gh_request(method: str, url: str, token: str, payload: Dict[str, Any] | None = None) -> Any:
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "TurnFocus-Artifact-Uploader",
    }
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub API {method} {url} failed ({exc.code}): {body}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload TurnFocus artifact tree to GitHub via REST API")
    parser.add_argument("--owner", default="xifezhao")
    parser.add_argument("--repo", default="TurnFocus")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--message", default="Clean TurnFocus artifact codebase and synthetic GPU sanity-check logs")
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        print("Error: GITHUB_TOKEN environment variable is required.", file=sys.stderr)
        sys.exit(1)

    root = Path(__file__).resolve().parent.parent
    api_base = f"https://api.github.com/repos/{args.owner}/{args.repo}"

    # 1. Fetch current HEAD commit and base tree on branch
    ref = gh_request("GET", f"{api_base}/git/ref/heads/{args.branch}", token)
    parent_sha = ref["object"]["sha"]
    parent_commit = gh_request("GET", f"{api_base}/git/commits/{parent_sha}", token)
    base_tree_sha = parent_commit["tree"]["sha"]
    print(f"Base HEAD on {args.owner}/{args.repo}@{args.branch}: {parent_sha[:10]}")

    skip_dirs = {".git", "__pycache__", ".pytest_cache", "build", "dist"}
    skip_suffixes = (".pyc", ".pyo", ".tar.gz", ".DS_Store")

    files_to_upload: List[Path] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if any(part in skip_dirs or part.endswith(".egg-info") for part in rel.parts):
            continue
        if rel.name.endswith(skip_suffixes):
            continue
        files_to_upload.append(p)

    print(f"Uploading {len(files_to_upload)} files as Git blobs...")
    tree_entries: List[Dict[str, str]] = []
    for idx, p in enumerate(files_to_upload, 1):
        rel_posix = p.relative_to(root).as_posix()
        raw_bytes = p.read_bytes()
        b64_content = base64.b64encode(raw_bytes).decode("ascii")
        blob = gh_request(
            "POST",
            f"{api_base}/git/blobs",
            token,
            {"content": b64_content, "encoding": "base64"},
        )
        mode = "100755" if os.access(p, os.X_OK) and rel_posix.endswith((".sh", ".py")) else "100644"
        tree_entries.append({"path": rel_posix, "mode": mode, "type": "blob", "sha": blob["sha"]})
        if idx % 10 == 0 or idx == len(files_to_upload):
            print(f"  [{idx}/{len(files_to_upload)}] uploaded {rel_posix}")

    # 2. Create new Git tree, commit, and fast-forward branch ref
    new_tree = gh_request(
        "POST",
        f"{api_base}/git/trees",
        token,
        {"base_tree": base_tree_sha, "tree": tree_entries},
    )
    new_commit = gh_request(
        "POST",
        f"{api_base}/git/commits",
        token,
        {"message": args.message, "tree": new_tree["sha"], "parents": [parent_sha]},
    )
    gh_request(
        "PATCH",
        f"{api_base}/git/refs/heads/{args.branch}",
        token,
        {"sha": new_commit["sha"], "force": False},
    )
    print(f"SUCCESS! Pushed commit {new_commit['sha'][:10]} to https://github.com/{args.owner}/{args.repo}")


if __name__ == "__main__":
    main()
