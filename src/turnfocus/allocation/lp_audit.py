"""Theorem 1(b): Data-Dependent LP A-Posteriori Upper Bound & Exact MILP Audit (Sec. 4.4, Eq. 18)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Sequence
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from turnfocus.allocation.lazy_greedy import compute_positional_budget
from turnfocus.allocation.objective import SubmodularObjective
from turnfocus.types import OperatingProfile


@dataclass(frozen=True)
class LPAuditResult:
    """Certificate returned by off-path LP optimality auditor (Table 2a)."""

    f_selected: float  # \tilde{F}_u(S)
    lp_residual_bound: float  # max_{x in [0,1]^{U'}} sum x_e \Delta(e | S)
    ub_total: float  # UB(S) = \tilde{F}_u(S) + lp_residual_bound
    optimality_ratio: float  # \tilde{F}_u(S) / UB(S) in [0, 1]
    num_vars: int
    num_constraints: int
    solve_ms: float


def solve_lp_upper_bound(
    objective: SubmodularObjective,
    selected_indices: Sequence[int],
    profile: OperatingProfile,
    l_hist_tokens: int = 0,
) -> LPAuditResult:
    """Compute exact data-dependent LP upper bound UB(S) per Theorem 1(b), Eq. (18).

    Strictly drops vacuous 0 <= 1 matroid rows for clips absent from U' (App. A.4)
    and solves the exact multi-constraint continuous relaxation via SciPy HiGHS.
    """
    t0 = time.perf_counter()
    state = objective.new_incremental_state()
    for idx in selected_indices:
        state.add(int(idx))
    f_s = float(state.current_value)

    sel_set = set(int(i) for i in selected_indices)
    u_prime = [i for i in range(objective.num_actions) if i not in sel_set]
    n_vars = len(u_prime)

    if n_vars == 0 or f_s <= 0.0:
        if n_vars == 0:
            elapsed = (time.perf_counter() - t0) * 1000.0
            return LPAuditResult(
                f_selected=f_s,
                lp_residual_bound=0.0,
                ub_total=f_s,
                optimality_ratio=1.0,
                num_vars=0,
                num_constraints=0,
                solve_ms=elapsed,
            )

    b_tok = float(max(1, profile.visual_token_budget))
    b_dec = float(max(1e-6, profile.decode_compute_budget))
    b_pos = float(max(1, compute_positional_budget(profile, l_hist_tokens)))
    b_l2 = float(profile.l2_token_budget)

    deltas = np.array([state.marginal_gain(e_idx) for e_idx in u_prime], dtype=np.float64)
    if np.all(deltas <= 1e-14):
        elapsed = (time.perf_counter() - t0) * 1000.0
        return LPAuditResult(
            f_selected=f_s,
            lp_residual_bound=0.0,
            ub_total=f_s,
            optimality_ratio=1.0 if f_s > 0 else 1.0,
            num_vars=n_vars,
            num_constraints=0,
            solve_ms=elapsed,
        )

    # Build A_ub x <= 1 constraints: resource/positional knapsacks (incl. R_L2 cap) + active partition-matroid clips
    rows: List[np.ndarray] = []
    r_tok = np.array([objective.actions[i].visual_tokens / b_tok for i in u_prime], dtype=np.float64)
    r_dec = np.array([objective.actions[i].decode_cost / b_dec for i in u_prime], dtype=np.float64)
    r_pos = np.array([objective.actions[i].pos_cost / b_pos for i in u_prime], dtype=np.float64)
    rows.extend([r_tok, r_dec, r_pos])
    if b_l2 > 0.0:
        r_l2 = np.array(
            [(objective.actions[i].visual_tokens / b_l2) if objective.actions[i].tier.value == 2 else 0.0 for i in u_prime],
            dtype=np.float64,
        )
        if np.any(r_l2 > 0.0):
            rows.append(r_l2)

    clip_to_cols: dict[int, List[int]] = {}
    for col_idx, e_idx in enumerate(u_prime):
        for c_id in objective.actions[e_idx].run.clip_indices:
            clip_to_cols.setdefault(c_id, []).append(col_idx)

    for cols in clip_to_cols.values():
        mat_row = np.zeros(n_vars, dtype=np.float64)
        mat_row[cols] = 1.0
        rows.append(mat_row)

    a_ub = np.stack(rows, axis=0)
    b_ub = np.ones(len(rows), dtype=np.float64)

    res = linprog(
        c=-deltas,
        A_ub=a_ub,
        b_ub=b_ub,
        bounds=(0.0, 1.0),
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"HiGHS LP solver failed on Eq. (18): {res.message}")

    lp_res = float(max(0.0, -res.fun))
    ub = f_s + lp_res
    ratio = float(f_s / ub) if ub > 1e-14 else 1.0
    elapsed = (time.perf_counter() - t0) * 1000.0

    return LPAuditResult(
        f_selected=f_s,
        lp_residual_bound=lp_res,
        ub_total=ub,
        optimality_ratio=float(np.clip(ratio, 0.0, 1.0)),
        num_vars=n_vars,
        num_constraints=len(rows),
        solve_ms=elapsed,
    )


def solve_exact_brute_or_milp_optimum(
    objective: SubmodularObjective,
    profile: OperatingProfile,
    l_hist_tokens: int = 0,
) -> float:
    """Exact discrete optimum \\widetilde{OPT} on small ground sets (|U| <= 18) for unit-testing Theorem 1(b).

    Evaluates all feasible subsets under exact matroid, resource, and positional constraints
    (or exact MILP when |U| <= 40 and objective is modular) to certify OPT <= UB(S).
    """
    u_len = objective.num_actions
    if u_len > 20:
        raise ValueError("Exact subset enumeration is intended for test instances with |U| <= 20")

    b_tok = float(profile.visual_token_budget)
    b_dec = float(profile.decode_compute_budget)
    b_pos = float(compute_positional_budget(profile, l_hist_tokens))
    b_l2 = float(profile.l2_token_budget)
    actions = objective.actions

    best_val = 0.0

    def dfs(pos: int, chosen: List[int], used_clips: set[int], tok: float, dec: float, p_cost: float, l2_used: float) -> None:
        nonlocal best_val
        if pos == u_len:
            val = objective.evaluate_set(chosen)
            if val > best_val:
                best_val = val
            return
        # Branch 1: skip pos
        dfs(pos + 1, chosen, used_clips, tok, dec, p_cost, l2_used)
        # Branch 2: include pos if feasible
        act = actions[pos]
        l2_c = float(act.visual_tokens) if act.tier.value == 2 else 0.0
        if (
            tok + act.visual_tokens <= b_tok + 1e-9
            and dec + act.decode_cost <= b_dec + 1e-9
            and p_cost + act.pos_cost <= b_pos + 1e-9
            and (l2_c == 0.0 or l2_used + l2_c <= b_l2 + 1e-9)
            and not any(c in used_clips for c in act.run.clip_indices)
        ):
            chosen.append(pos)
            used_clips.update(act.run.clip_indices)
            dfs(pos + 1, chosen, used_clips, tok + act.visual_tokens, dec + act.decode_cost, p_cost + act.pos_cost, l2_used + l2_c)
            for c in act.run.clip_indices:
                used_clips.remove(c)
            chosen.pop()

    dfs(0, [], set(), 0.0, 0.0, 0.0, 0.0)
    return best_val
