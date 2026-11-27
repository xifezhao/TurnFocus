"""Matched-budget retrieval and compression baselines from Section 5.3."""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from turnfocus.allocation.lazy_greedy import compute_positional_budget, lazy_greedy_allocate
from turnfocus.allocation.objective import ObjectiveConfig, SubmodularObjective
from turnfocus.belief.banded_filter import clamped_audiovisual_reweight
from turnfocus.candidates.nms_stratified import build_candidate_pool
from turnfocus.indexing.dig import DIGCalibrator
from turnfocus.types import (
    ContiguousRun,
    IntentPrediction,
    OperatingProfile,
    ResolutionTier,
    RunTierAction,
    SessionState,
    VideoIndex,
)


def _atomic_run_map(index: VideoIndex) -> Dict[int, ContiguousRun]:
    """Map clip_idx -> length-1 ContiguousRun."""
    out: Dict[int, ContiguousRun] = {}
    for g in index.runs:
        if g.length == 1:
            out[g.clip_indices[0]] = g
    return out


def _make_single_clip_action(
    act_id: int,
    run: ContiguousRun,
    tier: ResolutionTier,
    profile: OperatingProfile,
) -> RunTierAction:
    vis = run.length * profile.clip_tokens(tier)
    dec = run.length * profile.clip_decode_cost(tier)
    pos = vis + profile.l_ts + 1
    return RunTierAction(
        action_id=act_id,
        run=run,
        tier=tier,
        visual_tokens=vis,
        decode_cost=dec,
        pos_cost=pos,
    )


def uniform_fps_allocate(
    index: VideoIndex,
    profile: OperatingProfile,
    l_hist_tokens: int = 0,
) -> List[RunTierAction]:
    """Baseline (i): Uniform FPS Sampling across [0, M-1] at fixed R_L1 resolution (Sec. 5.3)."""
    m = index.num_clips
    if m == 0:
        return []
    b_tok = profile.visual_token_budget
    b_dec = profile.decode_compute_budget
    b_pos = compute_positional_budget(profile, l_hist_tokens)

    tier = ResolutionTier.R_L1
    per_tok = profile.clip_tokens(tier)
    per_dec = profile.clip_decode_cost(tier)
    per_pos = per_tok + profile.l_ts + 1

    max_k = min(
        m,
        int(b_tok // max(1, per_tok)),
        int(b_dec // max(1e-6, per_dec)),
        int(b_pos // max(1, per_pos)),
    )
    if max_k <= 0:
        return []

    chosen_clips = sorted(set(np.linspace(0, m - 1, num=max_k, dtype=int).tolist()))
    atomic_map = _atomic_run_map(index)
    return [
        _make_single_clip_action(k, atomic_map[c_idx], tier, profile)
        for k, c_idx in enumerate(chosen_clips)
        if c_idx in atomic_map
    ]


def topk_clip_asr_allocate(
    index: VideoIndex,
    s_audio_raw: np.ndarray,
    s_vis_raw: np.ndarray,
    profile: OperatingProfile,
    l_hist_tokens: int = 0,
) -> List[RunTierAction]:
    """Baseline (ii): Single-turn Top-k CLIP/ASR retrieval without belief, NMS, or tier adaptation."""
    m = index.num_clips
    if m == 0:
        return []
    uniform_prior = np.full(m, 1.0 / m, dtype=np.float64)
    scores, _ = clamped_audiovisual_reweight(
        b_pred=uniform_prior,
        s_audio_raw=s_audio_raw,
        s_vis_maxpool_raw=s_vis_raw,
        eta_0=0.5,
        t_soft=0.5,
        asr_density=index.asr_density,
        audio_conf=0.8,
    )
    order = np.argsort(-scores)
    atomic_map = _atomic_run_map(index)
    tier = ResolutionTier.R_L1

    b_tok = profile.visual_token_budget
    b_dec = profile.decode_compute_budget
    b_pos = compute_positional_budget(profile, l_hist_tokens)

    selected: List[RunTierAction] = []
    used_tok, used_dec, used_pos = 0, 0.0, 0
    for c_idx in order:
        c_int = int(c_idx)
        if c_int not in atomic_map:
            continue
        act = _make_single_clip_action(len(selected), atomic_map[c_int], tier, profile)
        if (
            used_tok + act.visual_tokens <= b_tok
            and used_dec + act.decode_cost <= b_dec + 1e-9
            and used_pos + act.pos_cost <= b_pos
        ):
            selected.append(act)
            used_tok += act.visual_tokens
            used_dec += act.decode_cost
            used_pos += act.pos_cost

    selected.sort(key=lambda a: a.run.start_sec)
    return selected


def static_token_pruning_allocate(
    index: VideoIndex,
    s_audio_raw: np.ndarray,
    s_vis_raw: np.ndarray,
    profile: OperatingProfile,
    retained_pool_state: Dict[str, Set[int]],
    l_hist_tokens: int = 0,
    keep_ratio: float = 0.35,
) -> List[RunTierAction]:
    """Baseline (iii): Static Visual Token Pruning (Sec. 5.3).

    Models irreversible Turn-1 pruning + spatial compression: at Turn 1, prunes (1 - keep_ratio)
    of clips conditioned on Turn-1 salience and compresses frames to R_L1; subsequent turns can
    only retrieve from the surviving unpruned clip subset at R_L1.
    """
    m = index.num_clips
    if m == 0:
        return []
    uniform_prior = np.full(m, 1.0 / m, dtype=np.float64)
    scores, _ = clamped_audiovisual_reweight(
        b_pred=uniform_prior,
        s_audio_raw=s_audio_raw,
        s_vis_maxpool_raw=s_vis_raw,
        eta_0=0.5,
        t_soft=0.5,
        asr_density=index.asr_density,
        audio_conf=0.8,
    )

    if "surviving_clips" not in retained_pool_state:
        k_survive = max(1, int(round(keep_ratio * m)))
        top_turn1 = np.argsort(-scores)[:k_survive]
        retained_pool_state["surviving_clips"] = set(int(i) for i in top_turn1)

    surviving = retained_pool_state["surviving_clips"]
    ranked_surviving = sorted(surviving, key=lambda idx: -float(scores[idx]))
    atomic_map = _atomic_run_map(index)
    tier = ResolutionTier.R_L1  # Fixed-ratio spatial token compression stays at low effective spatial tier

    b_tok = profile.visual_token_budget
    b_dec = profile.decode_compute_budget
    b_pos = compute_positional_budget(profile, l_hist_tokens)

    selected: List[RunTierAction] = []
    used_tok, used_dec, used_pos = 0, 0.0, 0
    for c_int in ranked_surviving:
        if c_int not in atomic_map:
            continue
        act = _make_single_clip_action(len(selected), atomic_map[c_int], tier, profile)
        if (
            used_tok + act.visual_tokens <= b_tok
            and used_dec + act.decode_cost <= b_dec + 1e-9
            and used_pos + act.pos_cost <= b_pos
        ):
            selected.append(act)
            used_tok += act.visual_tokens
            used_dec += act.decode_cost
            used_pos += act.pos_cost

    selected.sort(key=lambda a: a.run.start_sec)
    return selected


def pure_facility_location_allocate(
    index: VideoIndex,
    belief: np.ndarray,
    profile: OperatingProfile,
    session: SessionState,
    dig_calibrator: DIGCalibrator,
    l_hist_tokens: int = 0,
    m_cand: int = 64,
    theta_nms: float = 0.50,
) -> List[RunTierAction]:
    """Baseline (iv): Pure Facility-Location Summarization maximizing \\tilde{F}_sum alone (Sec. 5.3)."""
    sum_only_intent = IntentPrediction(
        u_drill=0.0,
        u_comp=0.0,
        u_jump=0.0,
        u_sum=1.0,
        c_conf=1.0,
        delta_rel=0,
        queried_entities=(),
    )
    cand_actions = build_candidate_pool(
        index=index,
        belief=belief,
        u_sum=1.0,
        profile=profile,
        m_cand=m_cand,
        theta_nms=theta_nms,
    )
    obj = SubmodularObjective(
        index=index,
        actions=cand_actions,
        belief=belief,
        intent=sum_only_intent,
        session=session,
        dig_calibrator=dig_calibrator,
        config=ObjectiveConfig(disable_dig=True),
    )
    _, selected_actions, _ = lazy_greedy_allocate(obj, profile=profile, l_hist_tokens=l_hist_tokens)
    return selected_actions


def offline_agentic_search_allocate(
    index: VideoIndex,
    s_audio_raw: np.ndarray,
    s_vis_raw: np.ndarray,
    profile: OperatingProfile,
    step_scorer_fn: Optional[callable] = None,
    l_hist_tokens: int = 0,
    num_rounds: int = 3,
) -> Tuple[List[RunTierAction], int]:
    """Baseline (v): Offline Agentic Search (multi-round coarse-to-fine exploration reference, Sec. 5.3).

    Iteratively explores coarse buckets then zooms into high-scoring local neighborhoods across
    `num_rounds` sequential LLM/evaluator calls. Returns (selected_actions, num_agent_calls).
    """
    m = index.num_clips
    if m == 0:
        return [], 0

    base_scores = 0.5 * s_audio_raw + 0.5 * s_vis_raw
    explored_scores = base_scores.copy()
    agent_calls = 0

    # Round 1: Coarse bucket probe (1 representative per bucket)
    bucket_reps = []
    for b in index.buckets:
        in_b = [
            c.clip_idx
            for c in index.clips
            if min(c.end_sec, b.end_sec) > max(c.start_sec, b.start_sec)
        ]
        if in_b:
            bucket_reps.append(in_b[len(in_b) // 2])

    for r_round in range(num_rounds):
        agent_calls += 1
        if r_round == 0 and step_scorer_fn is not None and bucket_reps:
            probe_vals = step_scorer_fn(np.array(bucket_reps, dtype=np.int32))
            for c_id, val in zip(bucket_reps, probe_vals):
                # Propagate coarse score to bucket neighborhood
                lo, hi = max(0, c_id - 2), min(m, c_id + 3)
                explored_scores[lo:hi] += 0.5 * float(val)
        elif step_scorer_fn is not None:
            top_focus = np.argsort(-explored_scores)[: min(8, m)]
            probe_vals = step_scorer_fn(top_focus.astype(np.int32))
            for c_id, val in zip(top_focus, probe_vals):
                explored_scores[int(c_id)] += float(val)

    order = np.argsort(-explored_scores)
    atomic_map = _atomic_run_map(index)
    # Use R_L2 on highest-ranked clip if profile permits, R_L1 on remainder
    allow_l2 = ResolutionTier.R_L2 in profile.allowed_tiers

    b_tok = profile.visual_token_budget
    b_dec = profile.decode_compute_budget
    b_pos = compute_positional_budget(profile, l_hist_tokens)

    selected: List[RunTierAction] = []
    used_tok, used_dec, used_pos = 0, 0.0, 0
    for rank_i, c_idx in enumerate(order):
        c_int = int(c_idx)
        if c_int not in atomic_map:
            continue
        tier = ResolutionTier.R_L2 if (allow_l2 and rank_i < 2) else ResolutionTier.R_L1
        act = _make_single_clip_action(len(selected), atomic_map[c_int], tier, profile)
        if (
            used_tok + act.visual_tokens > b_tok
            or used_dec + act.decode_cost > b_dec + 1e-9
            or used_pos + act.pos_cost > b_pos
        ):
            if tier == ResolutionTier.R_L2:
                act = _make_single_clip_action(len(selected), atomic_map[c_int], ResolutionTier.R_L1, profile)
            if (
                used_tok + act.visual_tokens > b_tok
                or used_dec + act.decode_cost > b_dec + 1e-9
                or used_pos + act.pos_cost > b_pos
            ):
                continue
        selected.append(act)
        used_tok += act.visual_tokens
        used_dec += act.decode_cost
        used_pos += act.pos_cost

    selected.sort(key=lambda a: a.run.start_sec)
    return selected, agent_calls
