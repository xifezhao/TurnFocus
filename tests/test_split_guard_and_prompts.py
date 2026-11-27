"""Tests for video-disjoint split guard, 2048-token history cap, Eq. 17 positional budget, and prompts."""

from __future__ import annotations

from turnfocus.allocation.lazy_greedy import compute_positional_budget
from turnfocus.datasets.split_guard import compress_bounded_history, verify_video_disjoint_splits
from turnfocus.models.prompt_templates import (
    format_llm_judge_prompt,
    format_retrieve_then_prefill_prompt,
    parse_judge_verdict,
    parse_mc_option_letter,
)
from turnfocus.types import OperatingProfile, ResolutionTier
from tests.test_submodularity_and_lp import _make_synthetic_video_index


def test_video_disjoint_split_guard():
    verify_video_disjoint_splits(["v1", "v2"], ["v3", "v4"], ["v5", "v6"])
    for bad_args in ((["v1", "v2"], ["v2", "v4"], ["v5"]), (["v1"], ["v3"], ["v3", "v5"])):
        raised = False
        try:
            verify_video_disjoint_splits(*bad_args)
        except ValueError as exc:
            assert "Video-disjoint split violation" in str(exc)
            raised = True
        assert raised, "Expected ValueError on overlapping video splits"


def test_bounded_history_compression_respects_2048_cap():
    long_history = [
        (f"Question {i} " + ("detail " * 80), f"Answer {i} " + ("explanation " * 80))
        for i in range(20)
    ]
    compressed, tok_count = compress_bounded_history(long_history, l_hist_max=2048)
    assert tok_count <= 2048
    assert len(compressed) < len(long_history)
    assert compressed[0][0] == "[Earlier Dialogue Summary]"


def test_eq17_positional_budget_equivalence():
    """Check sum_{j=1}^m n_{g_j}(tau_j) + L_hist + (m-1)(L_ts + 1) <= L_pos^max iff sum r_pos(e) <= B_pos."""
    profile = OperatingProfile(
        name="P1",
        visual_token_budget=512,
        decode_compute_budget=10.0,
        allowed_tiers=(ResolutionTier.R_L1, ResolutionTier.R_L2),
        l_pos_max=4096,
        l_ts=8,
    )
    l_hist = 600
    b_pos = compute_positional_budget(profile, l_hist)
    n_tokens = [64, 256, 128]
    m = len(n_tokens)
    lhs_raw = sum(n_tokens) + l_hist + (m - 1) * (profile.l_ts + 1)
    lhs_rpos = sum(n + profile.l_ts + 1 for n in n_tokens)
    assert (lhs_raw <= profile.l_pos_max) == (lhs_rpos <= b_pos)
    assert profile.l_pos_max - lhs_raw == b_pos - lhs_rpos


def test_verbatim_prompt_templates_and_parsers():
    idx = _make_synthetic_video_index(seed=3, num_clips=4, l_run=1)
    prompt = format_retrieve_then_prefill_prompt(
        index=idx,
        selected_actions=[],
        history=[("What is this lecture about?", "Submodular optimization.")],
        current_query="Which bound is derived in Section 4.4?",
        is_multiple_choice=True,
        options=["A. Chernoff bound", "B. HiGHS LP upper bound"],
    )
    assert "[System] You are a precise multimodal video assistant." in prompt
    assert "[Video Evidence (Chronological)]" in prompt
    assert "[Prior Conversation History]" in prompt
    assert "Respond with the single best option letter." in prompt
    assert parse_mc_option_letter("The best option is (B).") == "B"

    j_prompt = format_llm_judge_prompt("What is x?", "x = 42", "The equation shows x = 42.")
    assert '{"verdict": 0 or 1, "rationale": "<one concise sentence>"}' in j_prompt
    v, rat = parse_judge_verdict('{"verdict": 1, "rationale": "Exact match on x = 42."}')
    assert v == 1
    assert "42" in rat
