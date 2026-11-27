#!/usr/bin/env bash
# Builds a strictly anonymized, metadata-free zip archive for double-blind peer review
# (e.g., ACL ARR / ACM supplementary material upload or anonymous.4open.science).
# Strips .git history, OS metadata, GitHub uploader scripts, and any cached artifacts.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ZIP="${1:-${ROOT_DIR}/turnfocus_anonymous_submission.zip}"

TMP_STAGE="$(mktemp -d)"
trap 'rm -rf "${TMP_STAGE}"' EXIT

STAGE_DIR="${TMP_STAGE}/turnfocus_anonymous"
mkdir -p "${STAGE_DIR}"

COPYFILE_DISABLE=1 tar -C "${ROOT_DIR}" \
  --exclude='.git' \
  --exclude='.DS_Store' \
  --exclude='._*' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.pytest_cache' \
  --exclude='*.egg-info' \
  --exclude='scripts/upload_to_github.py' \
  --exclude='turnfocus_anonymous_submission.zip' \
  -cf - . | tar -C "${STAGE_DIR}" -xf -

(cd "${TMP_STAGE}" && zip -rq -X "${OUT_ZIP}" turnfocus_anonymous)
echo "[ANON_EXPORT_OK] Created anonymized double-blind archive: ${OUT_ZIP}"
