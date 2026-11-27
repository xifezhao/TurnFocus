#!/usr/bin/env bash
# Orchestrates end-to-end Google Colab GPU evaluation via the official `colab` CLI.
# Always stops the remote Colab runtime via `trap` on exit to prevent compute-unit leakage.
set -euo pipefail

GPU_TYPE="A100"
SESSION_NAME="turnfocus-ae"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU_TYPE="$2"; shift 2 ;;
    --session|-s) SESSION_NAME="$2"; shift 2 ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT_LOCAL="${ROOT_DIR}/results/colab_gpu_run"
mkdir -p "${OUT_LOCAL}"

cleanup() {
  echo "[colab-orchestrator] Releasing Colab runtime '${SESSION_NAME}'..."
  colab stop -s "${SESSION_NAME}" || true
}
trap cleanup EXIT

echo "[1/6] Provisioning Google Colab GPU session (${GPU_TYPE})..."
colab new --gpu "${GPU_TYPE}" -s "${SESSION_NAME}"

echo "[2/6] Packaging and uploading TurnFocus artifact..."
TAR_PATH="/tmp/turnfocus_artifact_upload.tar.gz"
tar -czf "${TAR_PATH}" -C "${ROOT_DIR}" src tests configs scripts pyproject.toml requirements-colab.txt README.md ARTIFACT_APPENDIX.md
colab upload -s "${SESSION_NAME}" "${TAR_PATH}" /content/turnfocus_artifact.tar.gz

echo "[3/6] Unpacking & installing dependencies on Colab VM..."
colab exec -s "${SESSION_NAME}" <<'PY'
import subprocess
subprocess.run(["mkdir", "-p", "/content/turnfocus"], check=True)
subprocess.run(["tar", "-xzf", "/content/turnfocus_artifact.tar.gz", "-C", "/content/turnfocus"], check=True)
PY
colab install -s "${SESSION_NAME}" -r "${ROOT_DIR}/requirements-colab.txt"

echo "[4/6] Executing full TurnFocus GPU Artifact Evaluation on Colab..."
colab exec -s "${SESSION_NAME}" <<'PY'
import subprocess, sys
res = subprocess.run(
    [sys.executable, "/content/turnfocus/scripts/run_colab_ae.py", "--output-dir", "/content/turnfocus/results"],
    cwd="/content/turnfocus",
)
if res.returncode != 0:
    raise SystemExit(res.returncode)
subprocess.run(["tar", "-czf", "/content/turnfocus_results.tar.gz", "-C", "/content/turnfocus/results", "."], check=True)
PY

echo "[5/6] Downloading measured results & exporting replayable Jupyter session log..."
colab download -s "${SESSION_NAME}" /content/turnfocus_results.tar.gz "${OUT_LOCAL}/turnfocus_results.tar.gz"
tar -xzf "${OUT_LOCAL}/turnfocus_results.tar.gz" -C "${OUT_LOCAL}"
colab log -s "${SESSION_NAME}" -o "${OUT_LOCAL}/colab_session_replay.ipynb"

echo "[6/6] Complete! Results saved to ${OUT_LOCAL}"
