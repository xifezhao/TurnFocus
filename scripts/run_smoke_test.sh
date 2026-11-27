#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

PY_BIN="${PYTHON_BIN:-python3}"

echo "=== [1/2] Running TurnFocus Mathematical & Functional Unit Tests ==="
if PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}" "${PY_BIN}" -c "import pytest" 2>/dev/null; then
  PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}" "${PY_BIN}" -m pytest tests/ -v
else
  PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}" "${PY_BIN}" - <<'PYTEST_FALLBACK'
import importlib, inspect
for mod_name in (
    "tests.test_submodularity_and_lp",
    "tests.test_belief_and_candidates",
    "tests.test_split_guard_and_prompts",
):
    mod = importlib.import_module(mod_name)
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        if name.startswith("test_"):
            fn()
            print(f"  PASSED {mod_name}::{name}")
PYTEST_FALLBACK
fi

echo "=== [2/2] Running TurnFocus End-to-End CPU Smoke Check (LP + Calibration + Guard) ==="
PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}" "${PY_BIN}" - <<'PY'
import numpy as np
from turnfocus.calibration.dev_calibrator import calibrate_on_dev_split
from turnfocus.datasets.split_guard import verify_video_disjoint_splits
from turnfocus.evaluation.pipeline import TurnFocusEngine, TurnFocusHyperparams
from turnfocus.types import IntentPrediction, OperatingProfile, ResolutionTier, SessionState
from tests.test_submodularity_and_lp import _make_synthetic_video_index

verify_video_disjoint_splits(["tr_1"], ["dev_1", "dev_2"], ["te_1", "te_2"])
dev_vids = [_make_synthetic_video_index(seed=s, num_clips=20, l_run=3) for s in (1, 2)]
te_vid = _make_synthetic_video_index(seed=99, num_clips=60, l_run=3)

p1 = OperatingProfile("P1", visual_token_budget=384, decode_compute_budget=8.5, allowed_tiers=(ResolutionTier.R_L1, ResolutionTier.R_L2))
rng = np.random.default_rng(0)
dev_turns = [
    (0, rng.dirichlet(np.ones(20)), IntentPrediction(0.2, 0.5, 0.1, 0.2, 0.9, 0, ("speaker_a",))),
    (1, rng.dirichlet(np.ones(20)), IntentPrediction(0.5, 0.1, 0.1, 0.3, 0.85, 1, ())),
]
dev_router_data = (
    ["Summarize the main milestones.", "Compare speaker_a across opening vs Q&A.", "What formula did he write right after that?"],
    [0, 1, 2],
    [3, 1, 0],
)
dev_dig_y = [1 if c.ocr_area_ratio > 0.4 else 0 for v in dev_vids for c in v.clips]

frozen, dig_cal, router = calibrate_on_dev_split(dev_vids, dev_turns, dev_router_data, dev_dig_y, p1)
hp = TurnFocusHyperparams(mu_b=frozen.mu_b, gamma_conf=frozen.gamma_conf)
engine = TurnFocusEngine(router=router, dig_calibrator=dig_cal, hp=hp)

sess = SessionState.initialize(te_vid.num_clips)
for q in ["Summarize the main milestones.", "Compare speaker_a opening vs Q&A.", "What formula did he write right after that?"]:
    s_aud = rng.normal(size=te_vid.num_clips)
    s_vis = rng.normal(size=te_vid.num_clips)
    alloc, lp = engine.step_turn(te_vid, sess, q, s_aud, s_vis, p1, run_lp_audit=True)
    assert lp is not None and 0.0 <= lp.optimality_ratio <= 1.0 + 1e-9
print(f"[SMOKE_OK] Frozen mu_B={frozen.mu_b:.4f}, LP ratio={lp.optimality_ratio:.4f}, online_ms={alloc.latency_ms['total_online_ms']:.2f}ms")
PY
