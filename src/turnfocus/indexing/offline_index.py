"""Non-destructive offline audiovisual index builder (Sec. 3.2)."""

from __future__ import annotations

import time
from typing import List, Optional
import numpy as np

from turnfocus.indexing.dig import DIGCalibrator
from turnfocus.indexing.segmentation import build_contiguous_runs, partition_temporal_buckets
from turnfocus.types import AtomicClip, VideoIndex


def build_offline_video_index(
    video_id: str,
    duration_sec: float,
    clips: List[AtomicClip],
    k_nbr: int = 32,
    l_run: int = 3,
    bucket_width_sec: float = 180.0,
    dig_calibrator: Optional[DIGCalibrator] = None,
) -> VideoIndex:
    """Build compact L0 + DIG index with pre-indexed sparse top-K_nbr demand neighbors.

    Note on neighbor orientation (Eq. 13 & App. A.4):
    In Eq. (13), \\tilde{sim}(v, c) = sim(v, c) * 1[v in N_{K_nbr}(c)], where c is the
    facility clip and v is the demand clip. Hence for each facility clip c, we store its
    top-K_nbr demand clips v ranked by sim(v, c) = max(0, cos(\\bar{z}_v, \\bar{z}_c)).
    """
    t0 = time.perf_counter()
    if len(clips) == 0:
        raise ValueError(f"Video {video_id} has 0 clips")

    cal = dig_calibrator or DIGCalibrator()
    spoken_duration = 0.0
    total_clip_duration = 0.0

    mean_embs = []
    for c in clips:
        c.phi_dig = cal.transform(c.ocr_area_ratio, c.hf_edge_density)
        dur = max(1e-3, c.duration_sec)
        total_clip_duration += dur
        if c.asr_text and len(c.asr_text.strip()) >= 3:
            spoken_duration += dur
        norm = np.linalg.norm(c.mean_embedding)
        mean_embs.append(c.mean_embedding / max(norm, 1e-12))

    asr_density = float(np.clip(spoken_duration / max(total_clip_duration, 1e-6), 0.0, 1.0))

    buckets = partition_temporal_buckets(duration_sec=duration_sec, bucket_width_sec=bucket_width_sec)
    runs = build_contiguous_runs(clips=clips, buckets=buckets, l_run=l_run)

    # Pairwise non-negative cosine similarity sim(v, c) = max(0, cos(\bar{z}_v, \bar{z}_c))
    z_mat = np.stack(mean_embs, axis=0).astype(np.float64)  # (M, D)
    sim_mat = np.maximum(0.0, z_mat @ z_mat.T)  # (M, M) where sim_mat[v, c] = sim(v, c)

    m = len(clips)
    k_eff = min(max(1, k_nbr), m)
    sparse_nbr_indices: List[np.ndarray] = []
    sparse_nbr_sims: List[np.ndarray] = []
    for c_idx in range(m):
        col = sim_mat[:, c_idx]
        if k_eff == m:
            top_v = np.argsort(-col)
        else:
            part = np.argpartition(-col, k_eff - 1)[:k_eff]
            top_v = part[np.argsort(-col[part])]
        pos_mask = col[top_v] > 0.0
        top_v = top_v[pos_mask]
        sparse_nbr_indices.append(top_v.astype(np.int32))
        sparse_nbr_sims.append(col[top_v].astype(np.float64))

    elapsed = time.perf_counter() - t0
    return VideoIndex(
        video_id=video_id,
        duration_sec=duration_sec,
        clips=clips,
        buckets=buckets,
        runs=runs,
        asr_density=asr_density,
        sparse_nbr_indices=sparse_nbr_indices,
        sparse_nbr_sims=sparse_nbr_sims,
        indexing_wall_sec=elapsed,
    )
