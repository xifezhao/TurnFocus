"""Online TurnFocus end-to-end grounding engine with stage-wise latency telemetry (Sec. 4)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import yaml

from turnfocus.allocation.lazy_greedy import lazy_greedy_allocate
from turnfocus.allocation.lp_audit import LPAuditResult, solve_lp_upper_bound
from turnfocus.allocation.objective import ObjectiveConfig, SubmodularObjective
from turnfocus.belief.banded_filter import banded_belief_predict, clamped_audiovisual_reweight
from turnfocus.belief.router import IntentCorefRouter
from turnfocus.belief.uncertainty_probe import ZeroConflictUncertaintyProbe
from turnfocus.candidates.nms_stratified import build_candidate_pool
from turnfocus.indexing.dig import DIGCalibrator
from turnfocus.types import (
    AllocationResult,
    OperatingProfile,
    SessionState,
    VideoIndex,
)


@dataclass
class TurnFocusHyperparams:
    """Complete online hyperparameter bundle matching Table 4 (tab:app_hyperparams)."""

    w_b: int = 3
    d_shift: int = 2
    eps_0: float = 0.05
    rho_shrink: float = 0.70
    gamma_conf: float = 0.80
    t_soft: float = 0.50
    eta_0: float = 0.60
    m_cand: int = 64
    theta_nms: float = 0.50
    gamma_red: float = 0.50
    beta_ent: float = 1.20
    k_a: float = 2.0
    mu_b: float = 0.15
    # Ablation switches (Table 3: i through vii)
    disable_nms: bool = False  # (i)
    disable_stratified: bool = False  # (ii)
    hard_intent_switching: bool = False  # (iii)
    unbucketed_entity: bool = False  # (iv-a)
    binary_entity_labels: bool = False  # (iv-b)
    disable_dig: bool = False  # (v)
    uniform_reset_ablation: bool = False  # (vi)
    fixed_unclamped_eta: bool = False  # (vii)
    force_visual_only_eta: bool = False  # Table 11 visual-only comparator

    @classmethod
    def from_yaml(cls, path: Path | str) -> "TurnFocusHyperparams":
        """Load hyperparameter defaults from a structured YAML config (e.g., configs/default_hyperparams.yaml)."""
        raw: Dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        bp = raw.get("belief_and_probe", {})
        cp = raw.get("candidate_pool", {})
        sa = raw.get("submodular_allocation", {})
        return cls(
            w_b=int(bp.get("w_b", cls.w_b)),
            d_shift=int(bp.get("d_shift", cls.d_shift)),
            eps_0=float(bp.get("eps_0", cls.eps_0)),
            rho_shrink=float(bp.get("rho_shrink", cls.rho_shrink)),
            gamma_conf=float(bp.get("gamma_conf", cls.gamma_conf)),
            t_soft=float(bp.get("t_soft", cls.t_soft)),
            eta_0=float(bp.get("eta_0", cls.eta_0)),
            m_cand=int(cp.get("m_cand", cls.m_cand)),
            theta_nms=float(cp.get("theta_nms", cls.theta_nms)),
            gamma_red=float(sa.get("gamma_red", cls.gamma_red)),
            beta_ent=float(sa.get("beta_ent", cls.beta_ent)),
            k_a=float(sa.get("k_a", cls.k_a)),
            mu_b=float(sa.get("mu_b", cls.mu_b)),
        )


def load_operating_profiles_from_yaml(path: Path | str) -> Dict[str, OperatingProfile]:
    """Load Operating Profiles (P0, P1, P2) from configs/default_hyperparams.yaml."""
    from turnfocus.types import ResolutionTier

    raw: Dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    profs_raw = raw.get("operating_profiles", {})
    tier_map = {"R_L1": ResolutionTier.R_L1, "R_L2": ResolutionTier.R_L2}
    out: Dict[str, OperatingProfile] = {}
    for p_name, cfg in profs_raw.items():
        tiers = tuple(tier_map[t] for t in cfg.get("allowed_tiers", ["R_L1"]))
        out[p_name] = OperatingProfile(
            name=p_name,
            visual_token_budget=int(cfg["visual_token_budget"]),
            decode_compute_budget=float(cfg["decode_compute_budget"]),
            allowed_tiers=tiers,
            tokens_per_clip_l1=int(cfg.get("tokens_per_clip_l1", 64)),
            tokens_per_clip_l2=int(cfg.get("tokens_per_clip_l2", 160)),
            max_l2_token_frac=float(cfg.get("max_l2_token_frac", 0.50)),
        )
    return out


@dataclass
class TurnFocusEngine:
    """Modular per-turn evidence grounding & resolution allocator (Figure 2)."""

    router: IntentCorefRouter = field(default_factory=IntentCorefRouter)
    dig_calibrator: DIGCalibrator = field(default_factory=DIGCalibrator)
    probe: ZeroConflictUncertaintyProbe = field(default_factory=ZeroConflictUncertaintyProbe)
    hp: TurnFocusHyperparams = field(default_factory=TurnFocusHyperparams)

    def step_turn(
        self,
        index: VideoIndex,
        session: SessionState,
        query: str,
        s_audio_raw: np.ndarray,
        s_vis_maxpool_raw: np.ndarray,
        profile: OperatingProfile,
        l_hist_tokens: int = 0,
        run_lp_audit: bool = False,
    ) -> tuple[AllocationResult, Optional[LPAuditResult]]:
        """Run one online turn of TurnFocus and optionally audit UB(S) off-path."""
        lat_ms: Dict[str, float] = {}

        # Stage 1: Intent & Coreference Router
        t0 = time.perf_counter()
        all_ents = {ent for c in index.clips for ent in c.entity_mentions.keys()}
        intent = self.router.predict(query=query, turn_idx=session.turn_idx, candidate_entities=all_ents)
        lat_ms["router_ms"] = (time.perf_counter() - t0) * 1000.0

        # Stage 2: O(M) Banded-Kernel Belief Filter & Clamped Audio-Visual Scoring
        t1 = time.perf_counter()
        if session.turn_idx == 0:
            b_pred = np.full(index.num_clips, 1.0 / index.num_clips, dtype=np.float64)
        else:
            b_pred = banded_belief_predict(
                b_prev=session.belief,
                intent=intent,
                w_b=self.hp.w_b,
                d_shift=self.hp.d_shift,
                eps_0=self.hp.eps_0,
                rho_shrink=self.hp.rho_shrink,
                gamma_conf=self.hp.gamma_conf,
                uniform_reset_ablation=self.hp.uniform_reset_ablation,
            )

        eff_eta_0 = 0.0 if self.hp.force_visual_only_eta else self.hp.eta_0
        b_coarse, eta_mod = clamped_audiovisual_reweight(
            b_pred=b_pred,
            s_audio_raw=s_audio_raw,
            s_vis_maxpool_raw=s_vis_maxpool_raw,
            eta_0=eff_eta_0,
            t_soft=self.hp.t_soft,
            asr_density=index.asr_density,
            audio_conf=intent.audio_conf,
            fixed_unclamped_ablation=(self.hp.fixed_unclamped_eta and not self.hp.force_visual_only_eta),
        )
        lat_ms["belief_l0_ms"] = (time.perf_counter() - t1) * 1000.0

        # Optional Conditional 2B Reranker (Zero-Conflict Uncertainty Probe)
        t_rerank = time.perf_counter()
        b_final, h_ret, triggered = self.probe.maybe_refine(query=query, belief=b_coarse)
        lat_ms["reranker_cond_ms"] = (time.perf_counter() - t_rerank) * 1000.0 if triggered else 0.0

        # Stage 3: 1D Temporal NMS + Stratified Bucket Candidate Pool
        t2 = time.perf_counter()
        cand_actions = build_candidate_pool(
            index=index,
            belief=b_final,
            u_sum=intent.u_sum,
            profile=profile,
            m_cand=self.hp.m_cand,
            theta_nms=self.hp.theta_nms,
            disable_nms=self.hp.disable_nms,
            disable_stratified=self.hp.disable_stratified,
        )
        lat_ms["nms_pool_ms"] = (time.perf_counter() - t2) * 1000.0

        # Stage 4: Smooth Intent-Weighted Submodular Lazy Greedy (Alg. 1)
        t3 = time.perf_counter()
        obj_cfg = ObjectiveConfig(
            gamma_red=self.hp.gamma_red,
            beta_ent=self.hp.beta_ent,
            k_a=self.hp.k_a,
            mu_b=self.hp.mu_b,
            hard_intent_switching=self.hp.hard_intent_switching,
            unbucketed_entity=self.hp.unbucketed_entity,
            binary_entity_labels=self.hp.binary_entity_labels,
            disable_dig=self.hp.disable_dig,
        )
        objective = SubmodularObjective(
            index=index,
            actions=cand_actions,
            belief=b_final,
            intent=intent,
            session=session,
            dig_calibrator=self.dig_calibrator,
            config=obj_cfg,
        )
        sel_indices, sel_actions, obj_val = lazy_greedy_allocate(
            objective=objective,
            profile=profile,
            l_hist_tokens=l_hist_tokens,
        )
        lat_ms["lazy_greedy_ms"] = (time.perf_counter() - t3) * 1000.0
        lat_ms["total_online_ms"] = (
            lat_ms["router_ms"]
            + lat_ms["belief_l0_ms"]
            + lat_ms["reranker_cond_ms"]
            + lat_ms["nms_pool_ms"]
            + lat_ms["lazy_greedy_ms"]
        )

        # Off-path LP Audit (strictly excluded from total_online_ms!)
        lp_cert: Optional[LPAuditResult] = None
        if run_lp_audit:
            lp_cert = solve_lp_upper_bound(
                objective=objective,
                selected_indices=sel_indices,
                profile=profile,
                l_hist_tokens=l_hist_tokens,
            )

        # Advance session state for Turn r+1
        session.belief = b_final
        session.update_grounded(sel_actions)
        session.turn_idx += 1

        res = AllocationResult(
            selected_actions=sel_actions,
            candidate_actions=cand_actions,
            objective_value=obj_val,
            belief=b_final,
            intent=intent,
            entropy_h_ret=h_ret,
            reranker_triggered=triggered,
            eta_mod=eta_mod,
            latency_ms=lat_ms,
        )
        return res, lp_cert
