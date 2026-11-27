"""Video timeline partitioning and bounded contiguous run construction (Sec. 3.1, Eq. 1)."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence
from turnfocus.types import AtomicClip, ContiguousRun, TemporalBucket


def temporal_iou(s1: float, e1: float, s2: float, e2: float) -> float:
    """Compute continuous 1D temporal IoU between [s1, e1] and [s2, e2] (Eq. 10)."""
    inter = max(0.0, min(e1, e2) - max(s1, s2))
    if inter <= 0.0:
        return 0.0
    union = max(e1, e2) - min(s1, s2)
    return inter / max(union, 1e-9)


def partition_temporal_buckets(
    duration_sec: float,
    bucket_width_sec: float = 180.0,
) -> List[TemporalBucket]:
    """Partition [0, T_video] into W coarse non-overlapping temporal buckets (Sec. 3.1)."""
    if duration_sec <= 0.0:
        raise ValueError("duration_sec must be > 0")
    if bucket_width_sec <= 0.0:
        raise ValueError("bucket_width_sec must be > 0")
    w_count = max(1, int(math.ceil(duration_sec / bucket_width_sec)))
    buckets: List[TemporalBucket] = []
    for w in range(w_count):
        start = w * bucket_width_sec
        end = min(duration_sec, (w + 1) * bucket_width_sec)
        if end > start:
            buckets.append(TemporalBucket(bucket_idx=w, start_sec=start, end_sec=end))
    return buckets


def build_contiguous_runs(
    clips: Sequence[AtomicClip],
    buckets: Sequence[TemporalBucket],
    l_run: int = 3,
) -> List[ContiguousRun]:
    """Construct bounded contiguous runs R = {(c_i, ..., c_j) | 1 <= j-i+1 <= L_run} (Eq. 1).

    Aggregates soft entity-mention confidences m_a(g) in [0, 1] across clips in g
    via noisy-OR 1 - prod_{c in g} (1 - m_a(c)), and records all coarse buckets I_w
    with positive interval intersection g \\cap I_w != \\emptyset.
    """
    if l_run < 1:
        raise ValueError("l_run must be >= 1")
    m = len(clips)
    runs: List[ContiguousRun] = []
    run_id = 0
    for i in range(m):
        max_j = min(m, i + l_run)
        for j in range(i, max_j):
            sub_clips = clips[i : j + 1]
            start_s = sub_clips[0].start_sec
            end_s = sub_clips[-1].end_sec

            # Noisy-OR soft entity confidence aggregation over clips in run g
            ent_non_detect: Dict[str, float] = {}
            for c in sub_clips:
                for ent, conf in c.entity_mentions.items():
                    clamped = float(min(1.0, max(0.0, conf)))
                    ent_non_detect[ent] = ent_non_detect.get(ent, 1.0) * (1.0 - clamped)
            ent_conf = {ent: 1.0 - nd for ent, nd in ent_non_detect.items() if (1.0 - nd) > 0.0}

            # Intersecting coarse temporal buckets (strictly positive overlap, or point containment)
            hit_buckets = tuple(
                b.bucket_idx
                for b in buckets
                if min(end_s, b.end_sec) > max(start_s, b.start_sec)
                or (start_s == end_s and b.start_sec <= start_s <= b.end_sec)
            )

            runs.append(
                ContiguousRun(
                    run_id=run_id,
                    clip_indices=tuple(range(i, j + 1)),
                    start_sec=start_s,
                    end_sec=end_s,
                    entity_confidences=ent_conf,
                    intersecting_buckets=hit_buckets,
                )
            )
            run_id += 1
    return runs
