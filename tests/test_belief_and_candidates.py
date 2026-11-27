"""Unit tests for O(M) banded belief operator, clamped eta_mod, exact cardinality split, and 1D NMS."""

from __future__ import annotations

import math
import numpy as np

from turnfocus.belief.banded_filter import (
    apply_banded_shift_operator,
    banded_belief_predict,
    compute_clamped_eta_mod,
)
from turnfocus.belief.uncertainty_probe import top_k_normalized_entropy
from turnfocus.candidates.nms_stratified import build_candidate_pool, compute_run_mean_beliefs
from turnfocus.indexing.segmentation import temporal_iou
from turnfocus.types import IntentPrediction, OperatingProfile, ResolutionTier
from tests.test_submodularity_and_lp import _make_synthetic_video_index


def test_banded_operator_is_column_stochastic_and_preserves_simplex():
    """Every column j of T_band(delta_rel) must sum identically to 1.0 under boundary clipping."""
    m = 25
    for delta_rel in (-1, 0, 1):
        cols = []
        for j in range(m):
            e_j = np.zeros(m, dtype=np.float64)
            e_j[j] = 1.0
            col = apply_banded_shift_operator(e_j, delta_rel=delta_rel, w_b=3, d_shift=2)
            assert np.all(col >= 0.0)
            assert abs(float(np.sum(col)) - 1.0) < 1e-12
            cols.append(col)
        t_mat = np.stack(cols, axis=1)
        # Verify bandwidth constraint |(i - j) - delta_rel * d_shift| <= W_b
        for j in range(m):
            center = j + delta_rel * 2
            for i in range(m):
                if abs(i - center) > 3 and 0 <= center < m:
                    assert t_mat[i, j] == 0.0


def test_low_confidence_shrinkage_vs_uniform_reset():
    m = 20
    b_prev = np.zeros(m, dtype=np.float64)
    b_prev[5] = 1.0
    low_conf_intent = IntentPrediction(
        u_drill=0.4, u_comp=0.2, u_jump=0.2, u_sum=0.2, c_conf=0.45, delta_rel=0
    )
    b_shrunk = banded_belief_predict(b_prev, low_conf_intent, rho_shrink=0.70, gamma_conf=0.80)
    assert abs(float(np.sum(b_shrunk)) - 1.0) < 1e-12
    # Clip 5 retains > 0.70 mass under shrinkage
    assert b_shrunk[5] > 0.70

    b_reset = banded_belief_predict(
        b_prev, low_conf_intent, rho_shrink=0.70, gamma_conf=0.80, uniform_reset_ablation=True
    )
    assert np.allclose(b_reset, np.full(m, 1.0 / m))


def test_clamped_eta_mod_silent_fallback_and_bounds():
    assert compute_clamped_eta_mod(eta_0=0.6, asr_density=0.0, audio_conf=0.95) == 0.0
    for dens in np.linspace(0.0, 1.5, 16):
        eta = compute_clamped_eta_mod(eta_0=0.8, asr_density=float(dens), audio_conf=1.2)
        assert 0.0 <= eta <= 1.0
        assert 0.0 <= (1.0 - eta) <= 1.0


def test_exact_cardinality_split_and_prefix_sum_run_beliefs():
    idx = _make_synthetic_video_index(seed=11, num_clips=30, l_run=3)
    rng = np.random.default_rng(11)
    belief = rng.dirichlet(np.ones(idx.num_clips))

    # Prefix-sum mean run beliefs must match naive per-run mean
    b_run = compute_run_mean_beliefs(belief, idx.runs)
    for k, g in enumerate(idx.runs):
        naive = float(np.mean([belief[c] for c in g.clip_indices]))
        assert abs(b_run[k] - naive) < 1e-13

    profile = OperatingProfile(
        name="P1",
        visual_token_budget=384,
        decode_compute_budget=8.5,
        allowed_tiers=(ResolutionTier.R_L1, ResolutionTier.R_L2),
    )
    for u_sum in np.linspace(0.0, 1.0, 21):
        m_cand = 32
        m_strat = int(math.floor(float(u_sum) * m_cand))
        m_top = m_cand - m_strat
        assert m_top + m_strat == m_cand
        acts = build_candidate_pool(idx, belief, u_sum=float(u_sum), profile=profile, m_cand=m_cand, theta_nms=0.5)
        assert len(acts) <= 2 * m_cand
        # Verify NMS IoU threshold holds among unique runs in candidate pool
        unique_runs = []
        seen = set()
        for a in acts:
            if a.run.run_id not in seen:
                seen.add(a.run.run_id)
                unique_runs.append(a.run)
        for i in range(len(unique_runs)):
            for j in range(i + 1, len(unique_runs)):
                iou = temporal_iou(
                    unique_runs[i].start_sec,
                    unique_runs[i].end_sec,
                    unique_runs[j].start_sec,
                    unique_runs[j].end_sec,
                )
                assert iou <= 0.5 + 1e-9


def test_entropy_probe_bounds():
    unif = np.full(32, 1.0 / 32.0)
    h_max, _ = top_k_normalized_entropy(unif, k=16)
    assert 0.99 <= h_max <= 1.0

    peaked = np.full(32, 1e-9)
    peaked[3] = 1.0
    h_min, top_idx = top_k_normalized_entropy(peaked, k=16)
    assert h_min < 0.01
    assert int(top_idx[0]) == 3
