"""O(M) column-stochastic banded-kernel belief filter & clamped audio-visual scoring (Sec. 4.1, Eqs. 4-7)."""

from __future__ import annotations

from typing import Tuple
import numpy as np

from turnfocus.types import IntentPrediction


def apply_banded_shift_operator(
    b_prev: np.ndarray,
    delta_rel: int,
    w_b: int = 3,
    d_shift: int = 2,
) -> np.ndarray:
    """Compute T_band(delta_rel) @ b_prev in exact O(W_b * M) = O(M) time (Sec. 4.1).

    Entry (i, j) is non-zero iff |(i - j) - delta_rel * d_shift| <= W_b, with
    triangular kernel weights (W_b + 1 - |offset|) and boundary-clipped column
    normalization so every column j of T_band sums identically to 1.0.
    """
    m = len(b_prev)
    if m == 0:
        raise ValueError("b_prev must be non-empty")
    if m == 1:
        return np.ones(1, dtype=np.float64)

    out = np.zeros(m, dtype=np.float64)
    shift = int(delta_rel) * int(d_shift)

    for j in range(m):
        mass = float(b_prev[j])
        if mass <= 0.0:
            continue
        center = j + shift
        lo = max(0, center - w_b)
        hi = min(m - 1, center + w_b)
        if lo > hi:
            # Boundary-clipped fallback onto nearest boundary clip
            target = 0 if center < 0 else (m - 1)
            out[target] += mass
            continue
        idx = np.arange(lo, hi + 1, dtype=np.int32)
        weights = (w_b + 1.0 - np.abs(idx - center)).astype(np.float64)
        w_sum = float(np.sum(weights))
        out[idx] += mass * (weights / w_sum)

    return out


def banded_belief_predict(
    b_prev: np.ndarray,
    intent: IntentPrediction,
    w_b: int = 3,
    d_shift: int = 2,
    eps_0: float = 0.05,
    rho_shrink: float = 0.70,
    gamma_conf: float = 0.80,
    uniform_reset_ablation: bool = False,
) -> np.ndarray:
    """Compute b_pred^(r) in O(M) time per Eq. (4).

    Args:
        uniform_reset_ablation: Implements Ablation (vi) in Table 3 — replaces gentle
            shrinkage with uniform 1/M reset when c_conf < gamma_conf.
    """
    m = len(b_prev)
    uniform = np.full(m, 1.0 / m, dtype=np.float64)

    if intent.c_conf < gamma_conf:
        if uniform_reset_ablation:
            return uniform
        b_pred = rho_shrink * b_prev + (1.0 - rho_shrink) * uniform
    else:
        p_jump = intent.u_jump + eps_0 * (1.0 - intent.u_jump)
        p_jump = float(np.clip(p_jump, eps_0, 1.0))
        shifted = apply_banded_shift_operator(b_prev, intent.delta_rel, w_b=w_b, d_shift=d_shift)
        b_pred = (1.0 - p_jump) * shifted + p_jump * uniform

    # Numerical safety: preserve simplex with strictly positive entries
    b_pred = np.maximum(b_pred, 1e-15)
    b_pred /= np.sum(b_pred)
    return b_pred


def compute_clamped_eta_mod(
    eta_0: float,
    asr_density: float,
    audio_conf: float,
    fixed_unclamped_ablation: bool = False,
) -> float:
    """Compute clamped audio modality weight eta_mod in [0, 1] per Eq. (7).

    Args:
        fixed_unclamped_ablation: Implements Ablation (vii) in Table 3 & Table 11 —
            fixes eta_mod = eta_0 without ASR_Density clamping.
    """
    if fixed_unclamped_ablation:
        return float(eta_0)
    raw = float(eta_0) * float(asr_density) * float(audio_conf)
    return float(np.clip(raw, 0.0, 1.0))


def _standardize(scores: np.ndarray) -> np.ndarray:
    std = float(np.std(scores))
    if std < 1e-8:
        return np.zeros_like(scores, dtype=np.float64)
    return (scores - float(np.mean(scores))) / std


def clamped_audiovisual_reweight(
    b_pred: np.ndarray,
    s_audio_raw: np.ndarray,
    s_vis_maxpool_raw: np.ndarray,
    eta_0: float = 0.60,
    t_soft: float = 0.50,
    asr_density: float = 1.0,
    audio_conf: float = 0.80,
    fixed_unclamped_ablation: bool = False,
) -> Tuple[np.ndarray, float]:
    """Reweight b_pred^(r) with standardized audio and max-pooled visual scores (Eqs. 5-7)."""
    if t_soft <= 0.0:
        raise ValueError("t_soft must be > 0")

    eta_mod = compute_clamped_eta_mod(
        eta_0=eta_0,
        asr_density=asr_density,
        audio_conf=audio_conf,
        fixed_unclamped_ablation=fixed_unclamped_ablation,
    )

    s_aud = _standardize(np.asarray(s_audio_raw, dtype=np.float64))
    s_vis = _standardize(np.asarray(s_vis_maxpool_raw, dtype=np.float64))

    fused = (eta_mod * s_aud + (1.0 - eta_mod) * s_vis) / t_soft
    # Log-domain Bayes-style update to prevent overflow/underflow
    log_unnorm = np.log(np.maximum(b_pred, 1e-15)) + fused
    log_unnorm -= np.max(log_unnorm)
    unnorm = np.exp(log_unnorm)
    b_post = unnorm / np.sum(unnorm)
    return b_post, eta_mod
