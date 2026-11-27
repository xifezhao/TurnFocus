"""Rigorous property tests for Lemma 1, Theorem 1(a) (norm/mono/submod), Alg. 1, and Theorem 1(b) LP bound."""

from __future__ import annotations

import numpy as np

from turnfocus.allocation.lazy_greedy import lazy_greedy_allocate
from turnfocus.allocation.lp_audit import solve_exact_brute_or_milp_optimum, solve_lp_upper_bound
from turnfocus.allocation.objective import ObjectiveConfig, SubmodularObjective
from turnfocus.candidates.nms_stratified import build_candidate_pool
from turnfocus.indexing.dig import DIGCalibrator
from turnfocus.indexing.offline_index import build_offline_video_index
from turnfocus.types import (
    AtomicClip,
    IntentPrediction,
    OperatingProfile,
    ResolutionTier,
    SessionState,
)


def _make_synthetic_video_index(seed: int = 42, num_clips: int = 10, l_run: int = 2):
    rng = np.random.default_rng(seed)
    clips = []
    for i in range(num_clips):
        kf = rng.normal(size=(3, 16))
        kf /= np.linalg.norm(kf, axis=1, keepdims=True)
        mean_e = kf.mean(axis=0)
        mean_e /= np.linalg.norm(mean_e)
        ents = {}
        if i % 3 == 0:
            ents["speaker_a"] = float(rng.uniform(0.3, 0.95))
        if i % 4 == 1:
            ents["topic_x"] = float(rng.uniform(0.2, 0.85))
        clips.append(
            AtomicClip(
                clip_idx=i,
                start_sec=i * 60.0,
                end_sec=(i + 1) * 60.0,
                asr_text=f"Clip {i} lecture explanation",
                speaker_id="spk_a",
                keyframe_embeddings=kf,
                mean_embedding=mean_e,
                ocr_area_ratio=float(rng.uniform(0.0, 0.8)),
                hf_edge_density=float(rng.uniform(0.02, 0.4)),
                entity_mentions=ents,
            )
        )
    return build_offline_video_index(
        video_id="vid_test",
        duration_sec=num_clips * 60.0,
        clips=clips,
        k_nbr=5,
        l_run=l_run,
        bucket_width_sec=180.0,
    )


def test_theorem1a_normalization_monotonicity_submodularity():
    """Verify \\tilde{F}_u(A) is normalized, monotone, and submodular over 300 random (A <= B, x) triples."""
    rng = np.random.default_rng(7)
    idx = _make_synthetic_video_index(seed=7, num_clips=12, l_run=3)
    profile = OperatingProfile(
        name="P1",
        visual_token_budget=512,
        decode_compute_budget=10.0,
        allowed_tiers=(ResolutionTier.R_L1, ResolutionTier.R_L2),
    )
    belief = rng.dirichlet(np.ones(idx.num_clips))
    intent = IntentPrediction(
        u_drill=0.25,
        u_comp=0.35,
        u_jump=0.10,
        u_sum=0.30,
        c_conf=0.90,
        delta_rel=1,
        queried_entities=("speaker_a", "topic_x"),
    )
    session = SessionState.initialize(idx.num_clips)
    session.grounded_tiers[2] = ResolutionTier.R_L1  # Test cross-turn upgrade weight w_fac!

    actions = build_candidate_pool(idx, belief, u_sum=intent.u_sum, profile=profile, m_cand=16)
    obj = SubmodularObjective(
        index=idx,
        actions=actions,
        belief=belief,
        intent=intent,
        session=session,
        dig_calibrator=DIGCalibrator(),
        config=ObjectiveConfig(mu_b=0.18, k_a=2.0),
    )

    # 1. Normalization
    assert abs(obj.evaluate_set([])) < 1e-15

    # 2. Monotonicity & Submodularity across 300 random nested subsets A <= B and x not in B
    u_size = obj.num_actions
    all_ids = np.arange(u_size)
    for _ in range(300):
        perm = rng.permutation(all_ids)
        size_a = int(rng.integers(0, max(1, u_size - 2)))
        size_b = int(rng.integers(size_a, max(size_a + 1, u_size - 1)))
        set_a = perm[:size_a].tolist()
        set_b = perm[:size_b].tolist()
        x = int(perm[size_b])

        f_a = obj.evaluate_set(set_a)
        f_b = obj.evaluate_set(set_b)
        f_ax = obj.evaluate_set(set_a + [x])
        f_bx = obj.evaluate_set(set_b + [x])

        assert f_b + 1e-12 >= f_a, f"Monotonicity violated: f(B)={f_b} < f(A)={f_a}"
        assert f_ax + 1e-12 >= f_a, "Marginal gain negative on A"
        assert f_bx + 1e-12 >= f_b, "Marginal gain negative on B"

        gain_a = f_ax - f_a
        gain_b = f_bx - f_b
        assert gain_a + 1e-11 >= gain_b, f"Submodularity violated: dA={gain_a} < dB={gain_b}"


def test_incremental_state_matches_exact_evaluation():
    """Verify O(L_run * K_nbr) IncrementalObjectiveState matches exact set evaluation step-by-step."""
    rng = np.random.default_rng(19)
    idx = _make_synthetic_video_index(seed=19, num_clips=14, l_run=3)
    profile = OperatingProfile(
        name="P2",
        visual_token_budget=768,
        decode_compute_budget=16.0,
        allowed_tiers=(ResolutionTier.R_L1, ResolutionTier.R_L2),
    )
    belief = rng.dirichlet(np.ones(idx.num_clips))
    intent = IntentPrediction(
        u_drill=0.20,
        u_comp=0.40,
        u_jump=0.05,
        u_sum=0.35,
        c_conf=0.92,
        delta_rel=0,
        queried_entities=("speaker_a",),
    )
    session = SessionState.initialize(idx.num_clips)
    actions = build_candidate_pool(idx, belief, u_sum=intent.u_sum, profile=profile, m_cand=20)
    obj = SubmodularObjective(
        index=idx,
        actions=actions,
        belief=belief,
        intent=intent,
        session=session,
        dig_calibrator=DIGCalibrator(),
        config=ObjectiveConfig(mu_b=0.20),
    )

    inc = obj.new_incremental_state()
    chosen = []
    for e_idx in rng.permutation(obj.num_actions)[:12]:
        e_int = int(e_idx)
        pred_gain = inc.marginal_gain(e_int)
        exact_before = obj.evaluate_set(chosen)
        exact_after = obj.evaluate_set(chosen + [e_int])
        assert abs(pred_gain - (exact_after - exact_before)) < 1e-11
        inc.add(e_int)
        chosen.append(e_int)
        assert abs(inc.current_value - exact_after) < 1e-11


def test_theorem1b_lp_upper_bound_dominates_exact_discrete_optimum():
    """Verify \\tilde{F}_u(S) <= \\widetilde{OPT} <= UB(S) on 20 random multi-constraint instances."""
    for trial in range(20):
        rng = np.random.default_rng(100 + trial)
        idx = _make_synthetic_video_index(seed=100 + trial, num_clips=7, l_run=2)
        profile = OperatingProfile(
            name="P1_small",
            visual_token_budget=320,
            decode_compute_budget=6.0,
            allowed_tiers=(ResolutionTier.R_L1, ResolutionTier.R_L2),
            l_pos_max=1024,
        )
        belief = rng.dirichlet(np.ones(idx.num_clips))
        u_raw = rng.dirichlet(np.ones(4))
        intent = IntentPrediction(
            u_drill=float(u_raw[0]),
            u_comp=float(u_raw[1]),
            u_jump=float(u_raw[2]),
            u_sum=float(u_raw[3]),
            c_conf=0.88,
            delta_rel=0,
            queried_entities=("speaker_a", "topic_x"),
        )
        session = SessionState.initialize(idx.num_clips)
        actions = build_candidate_pool(idx, belief, u_sum=intent.u_sum, profile=profile, m_cand=8)
        assert len(actions) <= 16

        obj = SubmodularObjective(
            index=idx,
            actions=actions,
            belief=belief,
            intent=intent,
            session=session,
            dig_calibrator=DIGCalibrator(),
            config=ObjectiveConfig(mu_b=0.15),
        )
        sel_ids, sel_acts, f_greedy = lazy_greedy_allocate(obj, profile=profile, l_hist_tokens=128)
        exact_opt = solve_exact_brute_or_milp_optimum(obj, profile=profile, l_hist_tokens=128)
        lp_cert = solve_lp_upper_bound(obj, sel_ids, profile=profile, l_hist_tokens=128)

        # Verify disjointness of lazy greedy output
        seen_clips = set()
        for act in sel_acts:
            for c_id in act.run.clip_indices:
                assert c_id not in seen_clips
                seen_clips.add(c_id)

        assert f_greedy <= exact_opt + 1e-10
        assert exact_opt <= lp_cert.ub_total + 1e-9, (
            f"Theorem 1(b) violated! exact_opt={exact_opt} > UB(S)={lp_cert.ub_total}"
        )
        assert 0.0 <= lp_cert.optimality_ratio <= 1.0 + 1e-9
