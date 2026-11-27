"""Strict video-disjoint offline calibration protocol (Appendix B.3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple
import numpy as np

from turnfocus.allocation.lazy_greedy import lazy_greedy_allocate
from turnfocus.allocation.objective import ObjectiveConfig, SubmodularObjective
from turnfocus.belief.router import IntentCorefRouter
from turnfocus.belief.uncertainty_probe import top_k_normalized_entropy
from turnfocus.candidates.nms_stratified import build_candidate_pool
from turnfocus.indexing.dig import DIGCalibrator
from turnfocus.types import IntentPrediction, OperatingProfile, SessionState, VideoIndex


@dataclass(frozen=True)
class FrozenCalibrationArtifact:
    """Immutable calibrated hyperparameters frozen strictly on dev split prior to test evaluation."""

    mu_b: float
    alpha_ocr: float
    tau_dig: float
    dig_roc_auc: float
    router_temperature: float
    gamma_conf: float
    router_dev_ece: float
    tau_h: float
    dev_reranker_trigger_rate: float


def calibrate_on_dev_split(
    dev_indices: Sequence[VideoIndex],
    dev_turn_beliefs: Sequence[Tuple[int, np.ndarray, IntentPrediction]],  # (vid_pos, b^(r), intent)
    dev_router_samples: Tuple[Sequence[str], Sequence[int], Sequence[int]],  # (queries, turn_idxs, labels)
    dev_dig_labels: Sequence[int],
    profile_p1: OperatingProfile,
    target_reranker_rate: float = 0.14,
    k_rerank_pool: int = 16,
) -> Tuple[FrozenCalibrationArtifact, DIGCalibrator, IntentCorefRouter]:
    """Execute all three calibration stages of Appendix B.3 strictly on video-disjoint dev data.

    1. Fit DIG alpha_ocr (maximizing ROC-AUC) & empirical CDF quantile map Calib(.).
    2. Calibrate offline-fixed belief scale mu_B > 0 as median captured belief mass on dev
       comparison turns at profile P_1 with beta_ent = 0 (strictly independent of online A!).
    3. Train & temperature-scale IntentCorefRouter (auditing dev ECE & gamma_conf) and
       choose entropy gate tau_H targeting <15% conditional 2B reranker trigger rate.
    """
    # Stage 2 (DIG Quantile Map) first so w_fac uses calibrated phi_dig
    all_areas: List[float] = []
    all_edges: List[float] = []
    for v_idx in dev_indices:
        for c in v_idx.clips:
            all_areas.append(c.ocr_area_ratio)
            all_edges.append(c.hf_edge_density)

    dig_cal = DIGCalibrator()
    dig_auc = dig_cal.fit_on_dev(
        dev_area_ocr=all_areas,
        dev_edge_hf=all_edges,
        dev_is_dense_labels=dev_dig_labels if len(dev_dig_labels) == len(all_areas) else None,
        dense_quantile_cutoff=0.70,
    )
    for v_idx in dev_indices:
        for c in v_idx.clips:
            c.phi_dig = dig_cal.transform(c.ocr_area_ratio, c.hf_edge_density)

    # Stage 1: Offline-Fixed Belief Scale mu_B on dev comparison turns with beta_ent = 0
    captured_masses: List[float] = []
    entropies: List[float] = []
    for vid_pos, belief, intent in dev_turn_beliefs:
        h_ret, _ = top_k_normalized_entropy(belief, k=k_rerank_pool)
        entropies.append(h_ret)
        if intent.u_comp >= 0.25 or len(dev_turn_beliefs) < 8:
            v_index = dev_indices[vid_pos]
            session = SessionState.initialize(v_index.num_clips)
            cands = build_candidate_pool(v_index, belief, u_sum=intent.u_sum, profile=profile_p1)
            obj_zero_ent = SubmodularObjective(
                index=v_index,
                actions=cands,
                belief=belief,
                intent=intent,
                session=session,
                dig_calibrator=dig_cal,
                config=ObjectiveConfig(beta_ent=0.0, mu_b=1.0),
            )
            sel_ids, _, _ = lazy_greedy_allocate(obj_zero_ent, profile=profile_p1)
            mass = float(sum(obj_zero_ent.v_drill[i] for i in sel_ids))
            if mass > 0.0:
                captured_masses.append(mass)

    mu_b = float(np.median(captured_masses)) if captured_masses else 0.15
    mu_b = max(mu_b, 1e-4)

    # Stage 3: Router confidence calibration & Entropy gate tau_H (<15% target)
    router = IntentCorefRouter()
    queries, turn_idxs, labels = dev_router_samples
    dev_ece, gamma_conf = router.fit_on_dev(queries=queries, turn_indices=turn_idxs, labels=labels)

    if entropies:
        # Conservative upper-tail quantile + small safety margin so test invocation rate stays strictly < 15%
        q_level = float(np.clip(1.0 - min(target_reranker_rate, 0.12), 0.85, 0.99))
        tau_h = float(min(0.995, np.quantile(entropies, q_level) + 0.025))
        realized_rate = float(np.mean(np.asarray(entropies) > tau_h))
    else:
        tau_h = 0.95
        realized_rate = 0.0

    frozen = FrozenCalibrationArtifact(
        mu_b=mu_b,
        alpha_ocr=dig_cal.alpha_ocr,
        tau_dig=dig_cal.tau_dig,
        dig_roc_auc=dig_auc,
        router_temperature=router.temperature,
        gamma_conf=gamma_conf,
        router_dev_ece=dev_ece,
        tau_h=tau_h,
        dev_reranker_trigger_rate=realized_rate,
    )
    return frozen, dig_cal, router
