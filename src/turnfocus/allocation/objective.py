"""Facility-consistent smooth intent-weighted submodular objective (Sec. 4.3, Eqs. 11-16)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import numpy as np

from turnfocus.indexing.dig import DIGCalibrator
from turnfocus.types import (
    IntentPrediction,
    ResolutionTier,
    RunTierAction,
    SessionState,
    VideoIndex,
)


@dataclass
class ObjectiveConfig:
    """Configuration and ablation flags for \\tilde{F}_u(A) (Sec. 4.3 & Table 3)."""

    gamma_red: float = 0.50
    beta_ent: float = 1.20
    k_a: float = 2.0
    mu_b: float = 0.15  # Offline-fixed belief scale strictly independent of A (App. B.3)
    hard_intent_switching: bool = False  # Ablation (iii)
    unbucketed_entity: bool = False  # Ablation (iv-a)
    binary_entity_labels: bool = False  # Ablation (iv-b)
    binary_entity_thresh: float = 0.50
    disable_dig: bool = False  # Ablation (v)


class SubmodularObjective:
    """Normalized, monotone, submodular objective \\tilde{F}_u(A) over U_cand(P) (Theorem 1a).

    Precomputes per-action modular drill gains, sparse demand facility coverage vectors,
    and entity-bucket confidence factors so both exact set evaluation and O(L_run * K_nbr)
    incremental lazy-greedy marginal evaluations match identically to floating-point precision.
    """

    def __init__(
        self,
        index: VideoIndex,
        actions: Sequence[RunTierAction],
        belief: np.ndarray,
        intent: IntentPrediction,
        session: SessionState,
        dig_calibrator: DIGCalibrator,
        config: ObjectiveConfig,
    ) -> None:
        self.index = index
        self.actions = list(actions)
        self.num_actions = len(self.actions)
        self.belief = np.asarray(belief, dtype=np.float64)
        self.intent = intent
        self.config = config

        if config.mu_b <= 0.0:
            raise ValueError("mu_b must be strictly > 0 and fixed offline")
        if not (0.0 <= config.gamma_red < 1.0):
            raise ValueError("gamma_red must be in [0, 1)")

        # Smooth vs. hard intent weights (Eq. 16 & Ablation iii)
        u_vec = np.array([intent.u_drill, intent.u_comp, intent.u_jump, intent.u_sum], dtype=np.float64)
        u_vec = np.maximum(u_vec, 0.0)
        u_sum_norm = float(np.sum(u_vec))
        u_vec = u_vec / u_sum_norm if u_sum_norm > 0 else np.array([0.25, 0.25, 0.25, 0.25])

        if config.hard_intent_switching:
            winner = int(np.argmax(u_vec))
            u_eff = np.zeros(4, dtype=np.float64)
            u_eff[winner] = 1.0
        else:
            u_eff = u_vec

        self.w_sum = float(u_eff[3])
        self.w_comp = float(u_eff[1])
        self.w_drill_total = float(u_eff[0] + u_eff[2] + u_eff[1])  # Note F_comp includes F_drill (Eq. 15)

        # Queried entities E_query that actually appear or are requested
        self.entities: Tuple[str, ...] = tuple(intent.queried_entities)
        self.num_entities = len(self.entities)
        self.num_buckets = 1 if config.unbucketed_entity else max(1, index.num_buckets)

        # Precompute facility-side weight w_fac(c, tau) for every clip and tier (Eq. 11)
        m = index.num_clips
        w_fac_l1 = np.empty(m, dtype=np.float64)
        w_fac_l2 = np.empty(m, dtype=np.float64)
        for c in index.clips:
            cid = c.clip_idx
            prev_t = session.prev_tier(cid)
            for tier, arr in ((ResolutionTier.R_L1, w_fac_l1), (ResolutionTier.R_L2, w_fac_l2)):
                q_t = dig_calibrator.tier_fidelity(
                    c.phi_dig,
                    tier_is_l2=(tier == ResolutionTier.R_L2),
                    disable_dig=config.disable_dig,
                )
                # Exempt genuine L1 -> L2 upgrades across turns!
                is_redundant = (prev_t != ResolutionTier.NONE) and (tier <= prev_t)
                disc = 1.0 - (config.gamma_red if is_redundant else 0.0)
                arr[cid] = q_t * disc

        # Precompute per-action modular drill score v_e and sparse demand coverage a_{v, e}
        self.v_drill = np.zeros(self.num_actions, dtype=np.float64)
        self.action_nbr_v: List[np.ndarray] = []
        self.action_nbr_val: List[np.ndarray] = []
        # Precompute (entity_idx, bucket_idx, one_minus_m_a) list per action
        self.action_ent_factors: List[List[Tuple[int, int, float]]] = []

        for e_idx, act in enumerate(self.actions):
            w_arr = w_fac_l2 if act.tier == ResolutionTier.R_L2 else w_fac_l1
            g = act.run

            # Eq. (12): v_e = sum_{c in g} b_c^(r) * w_fac(c, tau)
            self.v_drill[e_idx] = float(sum(self.belief[c_id] * w_arr[c_id] for c_id in g.clip_indices))

            # Eq. (13): a_{v, e} = max_{c in g} (w_fac(c, tau) * \tilde{sim}(v, c))
            demand_max: Dict[int, float] = {}
            for c_id in g.clip_indices:
                wc = float(w_arr[c_id])
                nbr_v = index.sparse_nbr_indices[c_id]
                nbr_s = index.sparse_nbr_sims[c_id]
                for v_id, s_val in zip(nbr_v, nbr_s):
                    prod_val = wc * float(s_val)
                    v_int = int(v_id)
                    if prod_val > demand_max.get(v_int, 0.0):
                        demand_max[v_int] = prod_val

            if demand_max:
                v_keys = np.fromiter(demand_max.keys(), dtype=np.int32, count=len(demand_max))
                v_vals = np.fromiter(demand_max.values(), dtype=np.float64, count=len(demand_max))
            else:
                v_keys = np.empty(0, dtype=np.int32)
                v_vals = np.empty(0, dtype=np.float64)
            self.action_nbr_v.append(v_keys)
            self.action_nbr_val.append(v_vals)

            # Eq. (14): Entity-bucket factors (1 - m_a(g))
            ent_list: List[Tuple[int, int, float]] = []
            if self.num_entities > 0:
                buckets_for_g = (0,) if config.unbucketed_entity else g.intersecting_buckets
                for a_idx, ent_name in enumerate(self.entities):
                    m_val = float(g.entity_confidences.get(ent_name, 0.0))
                    if config.binary_entity_labels:
                        m_val = 1.0 if m_val >= config.binary_entity_thresh else 0.0
                    m_val = float(np.clip(m_val, 0.0, 1.0))
                    if m_val > 0.0:
                        one_minus_m = 1.0 - m_val
                        for b_id in buckets_for_g:
                            if 0 <= b_id < self.num_buckets:
                                ent_list.append((a_idx, b_id, one_minus_m))
            self.action_ent_factors.append(ent_list)

        self.ent_scale = (
            (self.w_comp * config.beta_ent * config.mu_b) / (self.num_entities * config.k_a)
            if self.num_entities > 0
            else 0.0
        )

    def evaluate_set(self, action_indices: Sequence[int]) -> float:
        """Exact evaluation of \\tilde{F}_u(A) from scratch (used for verification & tests)."""
        if len(action_indices) == 0:
            return 0.0

        unique_e = sorted(set(int(i) for i in action_indices))
        # Drill component
        f_drill = float(np.sum(self.v_drill[unique_e]))

        # Sparse facility location component \tilde{F}_sum(A)
        f_sum = 0.0
        if self.w_sum > 0.0:
            m_cov = np.zeros(self.index.num_clips, dtype=np.float64)
            for e_idx in unique_e:
                v_ids = self.action_nbr_v[e_idx]
                v_vals = self.action_nbr_val[e_idx]
                if len(v_ids) > 0:
                    np.maximum.at(m_cov, v_ids, v_vals)
            f_sum = float(np.dot(self.belief, m_cov))

        # Entity bucket coverage component in F_comp(A)
        ent_bonus = 0.0
        if self.ent_scale > 0.0 and self.num_entities > 0:
            q_non_det = np.ones((self.num_entities, self.num_buckets), dtype=np.float64)
            for e_idx in unique_e:
                for a_idx, b_id, one_minus_m in self.action_ent_factors[e_idx]:
                    q_non_det[a_idx, b_id] *= one_minus_m
            h_a = np.sum(1.0 - q_non_det, axis=1)  # shape (|E_query|,)
            capped = np.minimum(self.config.k_a, h_a)
            ent_bonus = float(self.ent_scale * np.sum(capped))

        return float(self.w_drill_total * f_drill + self.w_sum * f_sum + ent_bonus)

    def new_incremental_state(self) -> "IncrementalObjectiveState":
        return IncrementalObjectiveState(self)


class IncrementalObjectiveState:
    """Maintains incremental state arrays (m_v, Q_{a,w}, current value) per Appendix A.4."""

    def __init__(self, obj: SubmodularObjective) -> None:
        self.obj = obj
        self.m_cov = np.zeros(obj.index.num_clips, dtype=np.float64)
        self.q_non_det = np.ones((max(1, obj.num_entities), obj.num_buckets), dtype=np.float64)
        self.h_a = np.zeros(max(1, obj.num_entities), dtype=np.float64)
        self.current_value: float = 0.0
        self.selected_set: set[int] = set()

    def marginal_gain(self, e_idx: int) -> float:
        """Compute \\Delta(e | A) = \\tilde{F}_u(A U {e}) - \\tilde{F}_u(A) in O(L_run * K_nbr) time."""
        if e_idx in self.selected_set:
            return 0.0
        obj = self.obj

        # 1. Modular drill gain
        gain = obj.w_drill_total * float(obj.v_drill[e_idx])

        # 2. Sparse facility location gain
        if obj.w_sum > 0.0:
            v_ids = obj.action_nbr_v[e_idx]
            v_vals = obj.action_nbr_val[e_idx]
            if len(v_ids) > 0:
                inc = np.maximum(0.0, v_vals - self.m_cov[v_ids])
                gain += obj.w_sum * float(np.dot(obj.belief[v_ids], inc))

        # 3. Soft temporal-bucket entity coverage gain
        if obj.ent_scale > 0.0 and obj.num_entities > 0:
            factors = obj.action_ent_factors[e_idx]
            if factors:
                # Aggregate per-(a, w) product in case g hits a bucket once
                delta_h: Dict[int, float] = {}
                for a_idx, b_id, one_minus_m in factors:
                    old_q = float(self.q_non_det[a_idx, b_id])
                    new_q = old_q * one_minus_m
                    delta_h[a_idx] = delta_h.get(a_idx, 0.0) + (old_q - new_q)
                k_cap = obj.config.k_a
                for a_idx, dh in delta_h.items():
                    old_cap = min(k_cap, float(self.h_a[a_idx]))
                    new_cap = min(k_cap, float(self.h_a[a_idx]) + dh)
                    gain += obj.ent_scale * (new_cap - old_cap)

        return float(max(0.0, gain))

    def add(self, e_idx: int) -> float:
        """Commit action e_idx to A and update incremental state in-place."""
        gain = self.marginal_gain(e_idx)
        if e_idx in self.selected_set:
            return 0.0
        obj = self.obj
        self.selected_set.add(e_idx)

        v_ids = obj.action_nbr_v[e_idx]
        v_vals = obj.action_nbr_val[e_idx]
        if len(v_ids) > 0:
            self.m_cov[v_ids] = np.maximum(self.m_cov[v_ids], v_vals)

        if obj.num_entities > 0:
            for a_idx, b_id, one_minus_m in obj.action_ent_factors[e_idx]:
                old_q = float(self.q_non_det[a_idx, b_id])
                new_q = old_q * one_minus_m
                self.q_non_det[a_idx, b_id] = new_q
                self.h_a[a_idx] += old_q - new_q

        self.current_value += gain
        return gain
