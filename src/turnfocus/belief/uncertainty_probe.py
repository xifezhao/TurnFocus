"""Zero-Conflict Uncertainty Probe & optional decoupled 2B MLLM reranker (Sec. 4.1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import numpy as np


def top_k_normalized_entropy(belief: np.ndarray, k: int = 16, eps: float = 1e-12) -> Tuple[float, np.ndarray]:
    """Compute normalized Shannon entropy H_ret in [0, 1] over renormalized top-K belief scores."""
    m = len(belief)
    k_eff = min(max(1, k), m)
    if k_eff == 1:
        return 0.0, np.array([0], dtype=np.int32)

    if k_eff == m:
        top_idx = np.argsort(-belief)
    else:
        part = np.argpartition(-belief, k_eff - 1)[:k_eff]
        top_idx = part[np.argsort(-belief[part])]

    top_vals = np.maximum(belief[top_idx], 0.0)
    total = float(np.sum(top_vals))
    if total <= 0.0:
        return 1.0, top_idx.astype(np.int32)

    b_tilde = top_vals / total
    entropy = -float(np.sum(b_tilde * np.log(b_tilde + eps)) / np.log(k_eff))
    return float(np.clip(entropy, 0.0, 1.0)), top_idx.astype(np.int32)


@dataclass
class ZeroConflictUncertaintyProbe:
    """Monitors H_ret > tau_H and conditionally invokes a decoupled 2B reranker (Sec. 4.1).

    Leaves the main backbone's fused FlashAttention kernels 100% untouched.
    """

    k_pool: int = 16
    tau_h: float = 0.88  # Calibrated on dev split to target <15% invocation rate
    blend_alpha: float = 0.50
    reranker_fn: Optional[Callable[[str, np.ndarray], np.ndarray]] = None

    def maybe_refine(
        self,
        query: str,
        belief: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool]:
        """Return (refined_belief, H_ret, triggered)."""
        h_ret, top_idx = top_k_normalized_entropy(belief, k=self.k_pool)
        if self.reranker_fn is None or h_ret <= self.tau_h:
            return belief, h_ret, False

        # Invoke decoupled 2B auxiliary probe on top-K clips only
        aux_scores = np.asarray(self.reranker_fn(query, top_idx), dtype=np.float64)
        if len(aux_scores) != len(top_idx):
            raise ValueError("reranker_fn must return one score per top_idx clip")

        # Softmax over the K probed clips preserving total top-K probability mass
        shifted = aux_scores - np.max(aux_scores)
        aux_probs = np.exp(shifted)
        aux_probs /= np.sum(aux_probs)

        refined = belief.copy()
        topk_mass = float(np.sum(refined[top_idx]))
        refined[top_idx] = (
            (1.0 - self.blend_alpha) * refined[top_idx]
            + self.blend_alpha * topk_mass * aux_probs
        )
        refined = np.maximum(refined, 1e-15)
        refined /= np.sum(refined)
        return refined, h_ret, True
