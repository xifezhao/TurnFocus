"""Ablation builders (Table 3) and one-at-a-time sensitivity sweep configurations (Table 6)."""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Tuple

from turnfocus.evaluation.pipeline import TurnFocusHyperparams


def build_ablation_variants(base_hp: TurnFocusHyperparams) -> List[Tuple[str, TurnFocusHyperparams]]:
    """Return exact 8 rows of Table 3 (tab:ablations): Full + (i) through (vii)."""
    return [
        ("TurnFocus (Full)", replace(base_hp)),
        ("(i) w/o 1D Temporal NMS", replace(base_hp, disable_nms=True)),
        ("(ii) w/o Stratified Summary Pool", replace(base_hp, disable_stratified=True)),
        ("(iii) Hard Intent Switching", replace(base_hp, hard_intent_switching=True)),
        ("(iv-a) Unbucketed Entity Coverage", replace(base_hp, unbucketed_entity=True)),
        ("(iv-b) Binary Entity Labels", replace(base_hp, binary_entity_labels=True)),
        ("(v) w/o DIG Fidelity Weighting", replace(base_hp, disable_dig=True)),
        ("(vi) Uniform Reset (c_conf < gamma_conf)", replace(base_hp, uniform_reset_ablation=True)),
        ("(vii) Fixed Unclamped eta_mod", replace(base_hp, fixed_unclamped_eta=True)),
    ]


def build_sensitivity_sweep_grid(base_hp: TurnFocusHyperparams) -> Dict[str, List[Tuple[str, dict]]]:
    """Return exact one-at-a-time sensitivity sweeps of Appendix B.4, Table 6 (tab:app_sweeps)."""
    return {
        "L_run": [
            ("L_run = 1 (atomic only)", {"l_run": 1, "hp": replace(base_hp)}),
            ("L_run = 2", {"l_run": 2, "hp": replace(base_hp)}),
            ("L_run = 3 (default)", {"l_run": 3, "hp": replace(base_hp)}),
            ("L_run = 4", {"l_run": 4, "hp": replace(base_hp)}),
        ],
        "M_cand": [
            ("M_cand = 16", {"m_cand": 16, "hp": replace(base_hp, m_cand=16)}),
            ("M_cand = 32", {"m_cand": 32, "hp": replace(base_hp, m_cand=32)}),
            ("M_cand = 64 (default)", {"m_cand": 64, "hp": replace(base_hp, m_cand=64)}),
            ("M_cand = 128", {"m_cand": 128, "hp": replace(base_hp, m_cand=128)}),
        ],
        "K_nbr": [
            ("K_nbr = 8", {"k_nbr": 8, "hp": replace(base_hp)}),
            ("K_nbr = 16", {"k_nbr": 16, "hp": replace(base_hp)}),
            ("K_nbr = 32 (default)", {"k_nbr": 32, "hp": replace(base_hp)}),
            ("K_nbr = 64", {"k_nbr": 64, "hp": replace(base_hp)}),
        ],
        "theta_nms": [
            ("theta_nms = 0.3", {"theta_nms": 0.3, "hp": replace(base_hp, theta_nms=0.3)}),
            ("theta_nms = 0.5 (default)", {"theta_nms": 0.5, "hp": replace(base_hp, theta_nms=0.5)}),
            ("theta_nms = 0.7", {"theta_nms": 0.7, "hp": replace(base_hp, theta_nms=0.7)}),
            ("theta_nms = 1.0 (no NMS)", {"theta_nms": 1.0, "hp": replace(base_hp, disable_nms=True)}),
        ],
    }
