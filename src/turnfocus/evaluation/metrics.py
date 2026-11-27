"""Six evaluation metrics, 95% CI statistics, and Cohen's kappa judge audit (Sec. 5.4)."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple
import numpy as np
from scipy import stats

from turnfocus.types import RunTierAction


def compute_temporal_grounding_iou(
    selected_actions: Sequence[RunTierAction],
    gt_intervals: Sequence[Tuple[float, float]],
) -> float:
    """Compute union-over-union 1D temporal IoU between retrieved runs A^(r) and annotated GT spans."""
    if not gt_intervals:
        return 0.0
    pred_spans = [(a.run.start_sec, a.run.end_sec) for a in selected_actions if a.run.end_sec > a.run.start_sec]
    if not pred_spans:
        return 0.0

    def merge_intervals(spans: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
        s_sorted = sorted((min(a, b), max(a, b)) for a, b in spans if b > a)
        if not s_sorted:
            return []
        merged = [s_sorted[0]]
        for cur_s, cur_e in s_sorted[1:]:
            last_s, last_e = merged[-1]
            if cur_s <= last_e + 1e-9:
                merged[-1] = (last_s, max(last_e, cur_e))
            else:
                merged.append((cur_s, cur_e))
        return merged

    m_pred = merge_intervals(pred_spans)
    m_gt = merge_intervals(gt_intervals)

    inter = 0.0
    for ps, pe in m_pred:
        for gs, ge in m_gt:
            inter += max(0.0, min(pe, ge) - max(ps, gs))

    len_pred = sum(e - s for s, e in m_pred)
    len_gt = sum(e - s for s, e in m_gt)
    union = len_pred + len_gt - inter
    return float(inter / max(union, 1e-9))


def mean_and_95ci(seed_values: Sequence[float]) -> Tuple[float, float]:
    """Compute sample mean and two-sided 95% Student-t confidence half-width across >= 5 seeds."""
    arr = np.asarray(seed_values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0
    mean = float(np.mean(arr))
    if n == 1:
        return mean, 0.0
    sd = float(np.std(arr, ddof=1))
    if sd < 1e-15:
        return mean, 0.0
    t_crit = float(stats.t.ppf(0.975, df=n - 1))
    half_width = t_crit * sd / math.sqrt(n)
    return mean, float(half_width)


def summarize_lp_optimality_distribution(ratios: Sequence[float]) -> Dict[str, float]:
    """Report Median, P10, and Worst-case ratio \\tilde{F}_u(S) / UB(S) (Sec. 5.4 & Table 2a)."""
    if not ratios:
        return {"median": 1.0, "p10": 1.0, "worst": 1.0}
    arr = np.asarray(ratios, dtype=np.float64)
    return {
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "worst": float(np.min(arr)),
    }


def summarize_latency_percentiles(ms_samples: Sequence[float]) -> Tuple[float, float]:
    """Return (P50_ms, P95_ms) for measured wall-clock component latencies (Table 2b)."""
    if not ms_samples:
        return 0.0, 0.0
    arr = np.asarray(ms_samples, dtype=np.float64)
    return float(np.percentile(arr, 50)), float(np.percentile(arr, 95))


def cohens_kappa_binary(rater_a: Sequence[int], rater_b: Sequence[int]) -> float:
    """Compute exact Cohen's kappa inter-annotator agreement on binary {0, 1} verdicts (Sec. 5.4)."""
    a = np.asarray(rater_a, dtype=np.int32)
    b = np.asarray(rater_b, dtype=np.int32)
    if len(a) == 0 or len(a) != len(b):
        raise ValueError("Rater arrays must be non-empty and equal length")
    p_o = float(np.mean(a == b))
    p_a1 = float(np.mean(a == 1))
    p_b1 = float(np.mean(b == 1))
    p_e = p_a1 * p_b1 + (1.0 - p_a1) * (1.0 - p_b1)
    if abs(1.0 - p_e) < 1e-12:
        return 1.0
    return float((p_o - p_e) / (1.0 - p_e))
