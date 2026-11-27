"""Dataset loaders, split discipline guards, and HuggingFace registry exports."""

from turnfocus.datasets.split_guard import (
    compress_bounded_history,
    estimate_token_count,
    tag_multiturn_stratum,
    verify_video_disjoint_splits,
)
from turnfocus.datasets.hf_benchmarks import (
    EvaluationTurn,
    HF_BENCHMARK_REGISTRY,
    VideoDialogueEpisode,
    fetch_hf_dataset_metadata_and_rows,
    render_highres_clip_frame,
)

__all__ = [
    "EvaluationTurn",
    "HF_BENCHMARK_REGISTRY",
    "VideoDialogueEpisode",
    "compress_bounded_history",
    "estimate_token_count",
    "fetch_hf_dataset_metadata_and_rows",
    "render_highres_clip_frame",
    "tag_multiturn_stratum",
    "verify_video_disjoint_splits",
]
