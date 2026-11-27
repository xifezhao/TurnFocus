"""Algorithm 1: Independent Per-Profile Lazy-Greedy Allocation (Sec. 4.3)."""

from __future__ import annotations

import heapq
from typing import List, Set, Tuple

from turnfocus.allocation.objective import SubmodularObjective
from turnfocus.types import OperatingProfile, RunTierAction


def compute_positional_budget(profile: OperatingProfile, l_hist_tokens: int) -> int:
    """Compute exact additive positional budget B_pos = L_pos^max - L_hist + L_ts + 1 (Eq. 17)."""
    l_hist_eff = min(max(0, l_hist_tokens), profile.l_hist_max)
    return max(0, profile.l_pos_max - l_hist_eff + profile.l_ts + 1)


def lazy_greedy_allocate(
    objective: SubmodularObjective,
    profile: OperatingProfile,
    l_hist_tokens: int = 0,
) -> Tuple[List[int], List[RunTierAction], float]:
    """Execute Algorithm 1 over U_cand(P) subject to partition matroid, resource, and B_pos budgets.

    Returns:
        (selected_action_indices_sorted_chronologically, selected_actions, final_objective_value)
    """
    b_tok = float(profile.visual_token_budget)
    b_dec = float(profile.decode_compute_budget)
    b_pos = float(compute_positional_budget(profile, l_hist_tokens))
    b_l2 = float(profile.l2_token_budget)

    if b_tok <= 0.0 or b_dec <= 0.0 or b_pos <= 0.0:
        return [], [], 0.0

    state = objective.new_incremental_state()
    actions = objective.actions

    # Max-heap via (-score, tie_break_action_id, step_evaluated, normalized_cost)
    heap: List[Tuple[float, int, int, float]] = []

    for e_idx, act in enumerate(actions):
        l2_cost = float(act.visual_tokens) if act.tier.value == 2 else 0.0
        if (
            act.visual_tokens > b_tok
            or act.decode_cost > b_dec
            or act.pos_cost > b_pos
            or (l2_cost > 0.0 and l2_cost > b_l2 + 1e-9)
        ):
            continue
        c_bar = max(
            act.visual_tokens / b_tok,
            act.decode_cost / b_dec,
            act.pos_cost / b_pos,
            (l2_cost / b_l2) if (l2_cost > 0.0 and b_l2 > 0.0) else 0.0,
        )
        c_bar = max(c_bar, 1e-12)
        singleton_gain = state.marginal_gain(e_idx)
        heapq.heappush(heap, (-(singleton_gain / c_bar), e_idx, 0, c_bar))

    cov_clips: Set[int] = set()
    used_tok = 0.0
    used_dec = 0.0
    used_pos = 0.0
    used_l2 = 0.0
    selected_indices: List[int] = []

    while heap:
        neg_score, e_idx, step_eval, c_bar = heapq.heappop(heap)
        act = actions[e_idx]
        l2_cost = float(act.visual_tokens) if act.tier.value == 2 else 0.0

        # Disjoint partition-matroid & simultaneous multi-budget feasibility check (Line 10 of Alg. 1)
        if any(c in cov_clips for c in act.run.clip_indices):
            continue
        if (
            used_tok + act.visual_tokens > b_tok + 1e-9
            or used_dec + act.decode_cost > b_dec + 1e-9
            or used_pos + act.pos_cost > b_pos + 1e-9
            or (l2_cost > 0.0 and used_l2 + l2_cost > b_l2 + 1e-9)
        ):
            continue

        if step_eval == len(selected_indices):
            # Fresh marginal gain (Line 13 of Alg. 1)
            state.add(e_idx)
            selected_indices.append(e_idx)
            cov_clips.update(act.run.clip_indices)
            used_tok += act.visual_tokens
            used_dec += act.decode_cost
            used_pos += act.pos_cost
            used_l2 += l2_cost
        else:
            delta = state.marginal_gain(e_idx)
            heapq.heappush(heap, (-(delta / c_bar), e_idx, len(selected_indices), c_bar))

    # Sort selected runs chronologically (Line 21 of Alg. 1)
    selected_indices.sort(key=lambda idx: (actions[idx].run.start_sec, actions[idx].run.end_sec, idx))
    selected_actions = [actions[idx] for idx in selected_indices]
    return selected_indices, selected_actions, float(state.current_value)
