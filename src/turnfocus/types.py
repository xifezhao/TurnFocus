"""Core typed data structures matching Section 3 and Section 4 of manuscript.tex."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple
import numpy as np


class ResolutionTier(IntEnum):
    """Strict resolution ordering: NONE < R_L1 < R_L2 (Sec. 3.3)."""

    NONE = 0
    R_L1 = 1
    R_L2 = 2

    @property
    def label(self) -> str:
        if self == ResolutionTier.R_L1:
            return "R_L1"
        if self == ResolutionTier.R_L2:
            return "R_L2"
        return "NONE"


@dataclass(frozen=True)
class OperatingProfile:
    """Request-level global operating profile P in {P_0, P_1, P_2} (Sec. 3.3).

    Attributes:
        name: Profile identifier ('P0', 'P1', or 'P2').
        visual_token_budget: Max visual tokens B_tok(P).
        decode_compute_budget: Max normalized decode/encode compute units B_dec(P).
        allowed_tiers: Permitted spatial resolution tiers T_res(P).
        l_pos_max: Hard backbone positional context cap L_pos^max.
        l_hist_max: Max dialogue history token cap L_hist^max (default 2048).
        l_ts: Token length of inter-run textual timestamp anchor L_ts.
        tokens_per_clip_l1: Visual token cost per atomic clip at R_L1.
        tokens_per_clip_l2: Visual token cost per atomic clip at R_L2.
    """

    name: str
    visual_token_budget: int
    decode_compute_budget: float
    allowed_tiers: Tuple[ResolutionTier, ...]
    l_pos_max: int = 16384
    l_hist_max: int = 2048
    l_ts: int = 8
    tokens_per_clip_l1: int = 64
    tokens_per_clip_l2: int = 160  # 2.5x R_L1 cost ratio (prevents R_L2 crowding out temporal breadth)
    max_l2_token_frac: float = 0.50  # Additive resource cap B_L2(P) = max_l2_token_frac * B_tok(P)

    @property
    def l2_token_budget(self) -> float:
        if ResolutionTier.R_L2 not in self.allowed_tiers:
            return 0.0
        return float(max(self.tokens_per_clip_l2, self.max_l2_token_frac * self.visual_token_budget))

    def clip_tokens(self, tier: ResolutionTier) -> int:
        if tier == ResolutionTier.R_L1:
            return self.tokens_per_clip_l1
        if tier == ResolutionTier.R_L2:
            return self.tokens_per_clip_l2
        raise ValueError(f"Unsupported tier for token counting: {tier}")

    def clip_decode_cost(self, tier: ResolutionTier) -> float:
        if tier == ResolutionTier.R_L1:
            return 1.0
        if tier == ResolutionTier.R_L2:
            return 2.4
        raise ValueError(f"Unsupported tier for decode cost: {tier}")


@dataclass
class AtomicClip:
    """Atomic audiovisual clip c_i spanning [start_sec, end_sec] (Sec. 3.1-3.2)."""

    clip_idx: int
    start_sec: float
    end_sec: float
    asr_text: str
    speaker_id: Optional[str]
    keyframe_embeddings: np.ndarray  # Shape (F_i, D), L2-normalized per keyframe
    mean_embedding: np.ndarray  # Shape (D,), L2-normalized \bar{z}_c
    ocr_area_ratio: float  # Area_ocr(c) in [0, 1]
    hf_edge_density: float  # Edge_hf(c) >= 0
    phi_dig: float = 0.0  # Calibrated DIG density \phi_dig(c) in [0, 1] (Eq. 2)
    entity_mentions: Dict[str, float] = field(default_factory=dict)  # Soft conf in [0, 1]
    keyframe_paths_or_indices: Tuple[int, ...] = ()

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


@dataclass(frozen=True)
class TemporalBucket:
    """Coarse non-overlapping temporal bucket I_w in T (Sec. 3.1)."""

    bucket_idx: int
    start_sec: float
    end_sec: float


@dataclass(frozen=True)
class ContiguousRun:
    """Bounded contiguous clip run g = (c_i, ..., c_j) with |g| <= L_run (Eq. 1)."""

    run_id: int
    clip_indices: Tuple[int, ...]
    start_sec: float
    end_sec: float
    entity_confidences: Dict[str, float]  # Soft m_a(g) in [0, 1]
    intersecting_buckets: Tuple[int, ...]  # Indices w where g \cap I_w != \emptyset

    @property
    def length(self) -> int:
        return len(self.clip_indices)

    @property
    def duration_sec(self) -> float:
        return max(1e-6, self.end_sec - self.start_sec)


@dataclass
class VideoIndex:
    """Non-destructive offline L0 + DIG audiovisual index for one video (Sec. 3.2)."""

    video_id: str
    duration_sec: float
    clips: List[AtomicClip]
    buckets: List[TemporalBucket]
    runs: List[ContiguousRun]
    asr_density: float  # ASR_Density(V) in [0, 1]
    # For each facility clip c, sparse_nbr_indices[c] gives demand clip indices v in N_{K_nbr}(c)
    # and sparse_nbr_sims[c] gives sim(v, c) = max(0, cos(\bar{z}_v, \bar{z}_c)) >= 0
    sparse_nbr_indices: List[np.ndarray]
    sparse_nbr_sims: List[np.ndarray]
    indexing_wall_sec: float = 0.0

    @property
    def num_clips(self) -> int:
        return len(self.clips)

    @property
    def num_buckets(self) -> int:
        return len(self.buckets)


@dataclass(frozen=True)
class IntentPrediction:
    """Router output at turn r (Sec. 4.1 & Table 8)."""

    u_drill: float
    u_comp: float
    u_jump: float
    u_sum: float
    c_conf: float
    delta_rel: int  # in {-1, 0, +1}
    queried_entities: Tuple[str, ...] = ()
    audio_conf: float = 0.8

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.u_drill, self.u_comp, self.u_jump, self.u_sum)


@dataclass(frozen=True)
class RunTierAction:
    """Action e = (g, \tau) in U_cand(P) with multi-resource & positional costs (Sec. 4.3)."""

    action_id: int
    run: ContiguousRun
    tier: ResolutionTier
    visual_tokens: int
    decode_cost: float
    pos_cost: int  # r_pos(g, \tau) = n_g(\tau) + L_ts + 1 (Eq. 17)


@dataclass
class SessionState:
    """Cross-turn belief and grounded clip resolution state (Sec. 3.3 & 4.1)."""

    turn_idx: int
    belief: np.ndarray  # b^(r) in \Delta^{M-1}
    grounded_tiers: Dict[int, ResolutionTier] = field(default_factory=dict)  # \mathcal{G}^{(r)}
    history: List[Tuple[str, str]] = field(default_factory=list)  # [(q^(1), y^(1)), ...]

    @classmethod
    def initialize(cls, num_clips: int) -> "SessionState":
        if num_clips <= 0:
            raise ValueError("num_clips must be positive")
        return cls(
            turn_idx=0,
            belief=np.full(num_clips, 1.0 / num_clips, dtype=np.float64),
            grounded_tiers={},
            history=[],
        )

    def prev_tier(self, clip_idx: int) -> ResolutionTier:
        return self.grounded_tiers.get(clip_idx, ResolutionTier.NONE)

    def update_grounded(self, selected: List[RunTierAction]) -> None:
        """Apply \tau_prev(c) <- max(\tau_prev(c), \tau_j) for all c in g_j (Sec. 3.3)."""
        for act in selected:
            for c_idx in act.run.clip_indices:
                prev = self.grounded_tiers.get(c_idx, ResolutionTier.NONE)
                if act.tier > prev:
                    self.grounded_tiers[c_idx] = act.tier


@dataclass
class AllocationResult:
    """Output of TurnFocus online allocation for turn r, plus latency telemetry."""

    selected_actions: List[RunTierAction]  # Chronologically ordered disjoint (g_j, \tau_j)
    candidate_actions: List[RunTierAction]  # U_cand(P)
    objective_value: float  # \tilde{F}_u(S)
    belief: np.ndarray  # Updated b^(r)
    intent: IntentPrediction
    entropy_h_ret: float
    reranker_triggered: bool
    eta_mod: float
    latency_ms: Dict[str, float]  # Stage-wise breakdown matching Table 2(b)
