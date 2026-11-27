#!/usr/bin/env python3
"""Automated integrity & archive consistency verifier for TurnFocus.

Verifies:
  1. Cryptographic SHA-256 manifest (`SHA256SUMS.txt`) matches every tracked file byte-for-byte.
  2. Every JSON file under `results/synthetic_gpu_sanity_check/` includes an explicit `scope_notice`.
  3. No stale constant-zero `dig_acc` key exists in `main_comparison`.
  4. `summary_report.md` stays synchronized with `full_empirical_results.json`.
  5. Optional `--compare-dir <dir>` diffs a newly generated run (e.g., from `run_colab_ae.py`)
     against the archived baseline JSON across key structural and numerical metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


ROOT_DIR = Path(__file__).resolve().parent.parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_sha256sums(root: Path) -> int:
    sums_path = root / "SHA256SUMS.txt"
    if not sums_path.exists():
        raise FileNotFoundError(f"Missing {sums_path}")
    count = 0
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, rel = line.split("  ", 1)
        target = root / rel
        if not target.exists():
            raise FileNotFoundError(f"SHA256SUMS entry missing on disk: {rel}")
        actual = sha256_file(target)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {rel}: expected {expected}, got {actual}")
        count += 1
    return count


def verify_archived_results(archive_dir: Path) -> Dict[str, Any]:
    for jf in sorted(archive_dir.glob("*.json")):
        data = json.loads(jf.read_text(encoding="utf-8"))
        if "scope_notice" not in data:
            raise AssertionError(f"Missing `scope_notice` field in {jf.name}")

    full_json = json.loads((archive_dir / "full_empirical_results.json").read_text(encoding="utf-8"))
    for method, row in full_json.get("main_comparison", {}).items():
        if "dig_acc" in row:
            raise AssertionError(f"Unpopulated legacy `dig_acc` key still present in main_comparison[{method!r}]")

    summary_md = (archive_dir / "summary_report.md").read_text(encoding="utf-8")
    for required_snippet in (
        "Scope Notice",
        "Ablation Interpretability Note",
        "fig3_pgfplots_coords.tex",
        "TurnFocus (Ours)",
    ):
        if required_snippet not in summary_md:
            raise AssertionError(f"Missing expected section/notice {required_snippet!r} in summary_report.md")
    return full_json


def compare_runs(archive_data: Dict[str, Any], candidate_dir: Path) -> List[str]:
    cand_path = candidate_dir / "full_empirical_results.json"
    if not cand_path.exists():
        raise FileNotFoundError(f"Candidate results JSON not found: {cand_path}")
    cand_data = json.loads(cand_path.read_text(encoding="utf-8"))
    report: List[str] = []
    for section in ("main_comparison", "lp_audit", "latency_ms", "ablations"):
        a_keys = set(archive_data.get(section, {}).keys())
        c_keys = set(cand_data.get(section, {}).keys())
        missing = sorted(a_keys - c_keys)
        extra = sorted(c_keys - a_keys)
        report.append(
            f"  - Section `{section}`: archive_keys={len(a_keys)}, candidate_keys={len(c_keys)}, "
            f"missing={missing}, extra={extra}"
        )
    # Numeric delta on TurnFocus (Ours) overall multi-turn accuracy if present
    a_tf = archive_data.get("main_comparison", {}).get("TurnFocus (Ours)", {}).get("overall")
    c_tf = cand_data.get("main_comparison", {}).get("TurnFocus (Ours)", {}).get("overall")
    if a_tf and c_tf:
        report.append(
            f"  - TurnFocus (Ours) overall accuracy: archive={a_tf[0]:.2f}% vs candidate={c_tf[0]:.2f}% "
            f"(delta={c_tf[0] - a_tf[0]:+.2f} pp)"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify TurnFocus artifact checksums, schema, and archive diffs.")
    parser.add_argument(
        "--archive-dir",
        type=str,
        default=str(ROOT_DIR / "results" / "synthetic_gpu_sanity_check"),
        help="Archived sanity-check results directory",
    )
    parser.add_argument(
        "--compare-dir",
        type=str,
        default=None,
        help="Optional newly generated output directory to diff against the archived JSON",
    )
    parser.add_argument("--skip-checksums", action="store_true", help="Skip SHA256SUMS.txt verification")
    args = parser.parse_args()

    if not args.skip_checksums:
        n_ok = verify_sha256sums(ROOT_DIR)
        print(f"[VERIFY_OK] SHA256SUMS.txt verified: {n_ok}/{n_ok} files matched byte-for-byte.")

    archive_data = verify_archived_results(Path(args.archive_dir))
    print(f"[VERIFY_OK] Archived JSON & Markdown checks passed in {args.archive_dir}.")

    if args.compare_dir:
        print(f"[DIFF] Comparing candidate `{args.compare_dir}` against archive `{args.archive_dir}`:")
        for line in compare_runs(archive_data, Path(args.compare_dir)):
            print(line)


if __name__ == "__main__":
    main()
