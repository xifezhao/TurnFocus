"""Video-disjoint split discipline, four-stratum tagging, and 2048-token history cap (Sec. 5.2 & App. C.2)."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Set, Tuple

from turnfocus.types import TemporalBucket


def verify_video_disjoint_splits(
    train_video_ids: Iterable[str],
    dev_video_ids: Iterable[str],
    test_video_ids: Iterable[str],
) -> None:
    """Strictly assert zero cross-split video leakage across Train, Dev, and Test (Sec. 5.2 & App. C.2)."""
    tr: Set[str] = set(train_video_ids)
    dv: Set[str] = set(dev_video_ids)
    te: Set[str] = set(test_video_ids)

    tr_dv = tr & dv
    tr_te = tr & te
    dv_te = dv & te
    if tr_dv or tr_te or dv_te:
        raise ValueError(
            f"Video-disjoint split violation! Overlaps: "
            f"train&dev={sorted(tr_dv)[:5]}, train&test={sorted(tr_te)[:5]}, dev&test={sorted(dv_te)[:5]}"
        )


def tag_multiturn_stratum(
    gt_intervals: Sequence[Tuple[float, float]],
    prev_grounded_intervals: Sequence[Tuple[float, float]],
    buckets: Sequence[TemporalBucket],
    targets_dig_segment: bool,
    asr_density: float,
    w_b_sec: float = 90.0,
    tau_silent: float = 0.20,
) -> List[str]:
    """Tag evaluation turn into the four analysis strata defined in Appendix C.2 Rule 2.

    Returns list of applicable tags from {'drill', 'jump', 'comp', 'silent', 'sum'}.
    """
    tags: List[str] = []
    if asr_density < tau_silent:
        tags.append("silent")

    # Bucket count spanned by GT evidence
    hit_buckets = set()
    for gs, ge in gt_intervals:
        for b in buckets:
            if min(ge, b.end_sec) > max(gs, b.start_sec):
                hit_buckets.add(b.bucket_idx)

    if len(hit_buckets) >= 2 and len(gt_intervals) >= 2:
        tags.append("comp")

    # Overlap / neighborhood relative to turn r-1
    overlaps_prev = False
    outside_local_wb = True
    if prev_grounded_intervals and gt_intervals:
        outside_local_wb = True
        for gs, ge in gt_intervals:
            for ps, pe in prev_grounded_intervals:
                if min(ge, pe + 15.0) >= max(gs, ps - 15.0):
                    overlaps_prev = True
                if min(ge, pe + w_b_sec) >= max(gs, ps - w_b_sec):
                    outside_local_wb = False
    else:
        outside_local_wb = False

    if targets_dig_segment or overlaps_prev:
        tags.append("drill")
    elif prev_grounded_intervals and outside_local_wb:
        tags.append("jump")

    if not tags:
        tags.append("sum" if len(hit_buckets) >= 3 else "drill")
    return tags


def estimate_token_count(text: str) -> int:
    """Deterministic wordpiece-scale token length estimator (~1.33 tokens/whitespace word)."""
    if not text:
        return 0
    return max(1, int(round(len(text.split()) * 1.35)))


def compress_bounded_history(
    history: Sequence[Tuple[str, str]],
    l_hist_max: int = 2048,
) -> Tuple[List[Tuple[str, str]], int]:
    """Deterministically bound cumulative Q/A history to <= L_hist^max tokens (App. C.2 Rule 4).

    Shared identically across TurnFocus and every baseline to prevent history-length confounds.
    """
    if not history:
        return [], 0

    turn_costs = [estimate_token_count(q) + estimate_token_count(a) + 4 for q, a in history]
    total = sum(turn_costs)
    if total <= l_hist_max:
        return list(history), total

    # Retain recent turns verbatim and deterministically summarize oldest prefix turns
    kept_recent: List[Tuple[str, str]] = []
    recent_tokens = 0
    split_idx = len(history)
    for idx in range(len(history) - 1, -1, -1):
        if recent_tokens + turn_costs[idx] + 96 <= l_hist_max:
            kept_recent.append(history[idx])
            recent_tokens += turn_costs[idx]
            split_idx = idx
        else:
            break
    kept_recent.reverse()

    older = history[:split_idx]
    if older:
        digest_parts = [f"T{i+1}: Q={q[:48]} -> A={a[:48]}" for i, (q, a) in enumerate(older)]
        summary_str = " ; ".join(digest_parts)
        words = summary_str.split()
        max_words = max(12, int((l_hist_max - recent_tokens - 12) / 1.35))
        summary_str = " ".join(words[:max_words])
        prefix_pair = ("[Earlier Dialogue Summary]", summary_str)
        final_hist = [prefix_pair] + kept_recent
    else:
        final_hist = kept_recent

    final_tokens = min(
        l_hist_max,
        sum(estimate_token_count(q) + estimate_token_count(a) + 4 for q, a in final_hist),
    )
    return final_hist, final_tokens
