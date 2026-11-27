"""Exact-cardinality 1D temporal NMS and stratified temporal-bucket candidate pooling (Sec. 4.2)."""

from __future__ import annotations

import math
from typing import List, Set
import numpy as np

from turnfocus.indexing.segmentation import temporal_iou
from turnfocus.types import ContiguousRun, OperatingProfile, RunTierAction, VideoIndex


def compute_run_mean_beliefs(belief: np.ndarray, runs: List[ContiguousRun]) -> np.ndarray:
    """Compute length-normalized mean belief b_run(g) in O(1) per run via 1D prefix sums (Eq. 8 & App. A.4)."""
    prefix = np.zeros(len(belief) + 1, dtype=np.float64)
    np.cumsum(belief, out=prefix[1:])
    scores = np.empty(len(runs), dtype=np.float64)
    for idx, g in enumerate(runs):
        i = g.clip_indices[0]
        j = g.clip_indices[-1]
        scores[idx] = (prefix[j + 1] - prefix[i]) / float(j - i + 1)
    return scores


def build_candidate_pool(
    index: VideoIndex,
    belief: np.ndarray,
    u_sum: float,
    profile: OperatingProfile,
    m_cand: int = 64,
    theta_nms: float = 0.50,
    disable_nms: bool = False,
    disable_stratified: bool = False,
) -> List[RunTierAction]:
    """Construct reduced run pool R_cand = R_top U R_strat (|R_cand| <= M_cand) and action ground set U_cand(P).

    Implements Eqs. (8)-(10) and supports Ablation (i) `disable_nms` (theta_nms = 1.0)
    and Ablation (ii) `disable_stratified` (M_strat = 0).
    """
    runs = index.runs
    if len(runs) == 0 or m_cand <= 0:
        return []

    b_run = compute_run_mean_beliefs(belief, runs)
    order = np.argsort(-b_run, kind="mergesort")

    u_sum_clamped = 0.0 if disable_stratified else float(np.clip(u_sum, 0.0, 1.0))
    m_strat = int(math.floor(u_sum_clamped * m_cand))
    m_top = int(m_cand - m_strat)
    assert m_top + m_strat == m_cand, "Exact cardinality split violated"

    eff_theta = 1.01 if disable_nms else float(theta_nms)

    selected_run_ids: Set[int] = set()
    r_top: List[ContiguousRun] = []

    # 1. Peak-focused 1D Temporal NMS subset R_top (Eq. 10)
    for r_idx in order:
        if len(r_top) >= m_top:
            break
        cand = runs[int(r_idx)]
        suppressed = False
        if not disable_nms and eff_theta < 1.0:
            for kept in r_top:
                if temporal_iou(cand.start_sec, cand.end_sec, kept.start_sec, kept.end_sec) > eff_theta:
                    suppressed = True
                    break
        if not suppressed:
            r_top.append(cand)
            selected_run_ids.add(cand.run_id)

    # 2. Stratified coarse temporal-bucket subset R_strat (Sec. 4.2)
    r_strat: List[ContiguousRun] = []
    w_total = index.num_buckets
    if m_strat > 0 and w_total > 0:
        # Group runs intersecting each bucket in descending b_run order
        bucket_queues: List[List[ContiguousRun]] = [[] for _ in range(w_total)]
        for r_idx in order:
            cand = runs[int(r_idx)]
            for b_id in cand.intersecting_buckets:
                if 0 <= b_id < w_total:
                    bucket_queues[b_id].append(cand)

        bucket_ptrs = [0] * w_total
        if w_total >= m_strat:
            # Evenly spaced coarse buckets across [0, W-1]
            target_buckets = [int(round(k * (w_total - 1) / max(1, m_strat - 1))) for k in range(m_strat)]
            if m_strat == 1:
                target_buckets = [w_total // 2]
        else:
            # Round-robin passes across all W buckets when W < M_strat
            target_buckets = [k % w_total for k in range(m_strat * 2)]

        for b_id in target_buckets:
            if len(r_strat) >= m_strat:
                break
            q = bucket_queues[b_id]
            while bucket_ptrs[b_id] < len(q):
                cand = q[bucket_ptrs[b_id]]
                bucket_ptrs[b_id] += 1
                if cand.run_id in selected_run_ids:
                    continue
                # Enforce NMS validity against already accepted runs if NMS enabled
                if not disable_nms and eff_theta < 1.0:
                    if any(
                        temporal_iou(cand.start_sec, cand.end_sec, kept.start_sec, kept.end_sec) > eff_theta
                        for kept in (r_top + r_strat)
                    ):
                        continue
                r_strat.append(cand)
                selected_run_ids.add(cand.run_id)
                break

    r_cand = r_top + r_strat
    assert len(r_cand) <= m_cand

    # Expand across resolution tiers permitted by global profile P: |U_cand(P)| <= 2 * M_cand
    actions: List[RunTierAction] = []
    act_id = 0
    for g in r_cand:
        for tier in profile.allowed_tiers:
            vis_toks = g.length * profile.clip_tokens(tier)
            dec_cost = g.length * profile.clip_decode_cost(tier)
            pos_cost = vis_toks + profile.l_ts + 1  # Eq. (17)
            actions.append(
                RunTierAction(
                    action_id=act_id,
                    run=g,
                    tier=tier,
                    visual_tokens=vis_toks,
                    decode_cost=dec_cost,
                    pos_cost=pos_cost,
                )
            )
            act_id += 1
    return actions
