"""Lightweight Intent & Temporal Coreference Router with temperature-scaled confidence (Sec. 4.1 & App. C.3)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple
import numpy as np

from turnfocus.types import IntentPrediction

_FORWARD_CUES = (
    "right after",
    "immediately after",
    "just after",
    "next slide",
    "following that",
    "after that",
    "what did he write next",
    "what happened next",
    "subsequently",
)
_BACKWARD_CUES = (
    "right before",
    "just before",
    "immediately before",
    "prior to that",
    "before that",
    "previous slide",
    "earlier right before",
)
_SUMMARY_CUES = (
    "summarize",
    "summary",
    "main milestones",
    "overview",
    "overall",
    "throughout the video",
    "key takeaways",
    "chronological outline",
    "how many times in total",
    "entire lecture",
)
_COMPARISON_CUES = (
    "compare",
    "comparison",
    "contrast",
    "difference between",
    "stance change",
    "how did her",
    "how did his",
    "opening to q&a",
    "opening vs",
    "both appearances",
    "across the two",
    "changed from",
)
_JUMP_CUES = (
    "later in the",
    "earlier in the",
    "switch to",
    "in another part",
    "at the end of the broadcast",
    "back at the beginning",
    "meanwhile",
    "unrelated",
)
_DRILL_CUES = (
    "formula",
    "equation",
    "whiteboard",
    "slide",
    "exact number",
    "read the",
    "transcribe",
    "chart",
    "table value",
    "small text",
    "zoom in",
    "right after that",
    "right before that",
    "detail",
)
_AUDIO_QUERY_CUES = (
    "say",
    "said",
    "mention",
    "speak",
    "speaker",
    "quote",
    "explain",
    "argue",
    "ask",
    "discuss",
    "stance",
)


def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / max(temperature, 1e-6)
    shifted = scaled - np.max(scaled)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


@dataclass
class IntentCorefRouter:
    """Four-way intent & coreference router calibrated via dev temperature scaling (Table 8 & App. B.3).

    Supports both learned linear head over text embeddings (when trained on dev split)
    and deterministic lexical-discourse feature scoring, calibrated via temperature scaling.
    """

    temperature: float = 0.85
    gamma_conf: float = 0.80
    weights: Optional[np.ndarray] = None  # Optional (F_dim, 4) linear head trained on dev split
    bias: Optional[np.ndarray] = None  # Optional (4,)
    known_entities: List[str] = field(default_factory=list)

    def extract_discourse_features(self, query: str, turn_idx: int) -> np.ndarray:
        """Extract 10D normalized discourse-cue feature vector for [drill, comp, jump, sum]."""
        q = query.lower().strip()
        f_drill = sum(1.0 for p in _DRILL_CUES if p in q)
        f_comp = sum(1.0 for p in _COMPARISON_CUES if p in q)
        f_jump = sum(1.0 for p in _JUMP_CUES if p in q)
        f_sum = sum(1.0 for p in _SUMMARY_CUES if p in q)
        has_pronoun = float(bool(re.search(r"\b(that|it|this|he|she|they|there|then)\b", q)))
        is_first_turn = 1.0 if turn_idx == 0 else 0.0
        f_fwd = float(any(p in q for p in _FORWARD_CUES))
        f_bwd = float(any(p in q for p in _BACKWARD_CUES))
        word_len = min(1.0, len(q.split()) / 25.0)
        has_question_what_exact = float(bool(re.search(r"\b(exact|specific|value|formula|write|wrote)\b", q)))
        return np.array(
            [
                f_drill,
                f_comp,
                f_jump,
                f_sum,
                has_pronoun,
                is_first_turn,
                f_fwd,
                f_bwd,
                word_len,
                has_question_what_exact,
            ],
            dtype=np.float64,
        )

    def fit_on_dev(
        self,
        queries: Sequence[str],
        turn_indices: Sequence[int],
        labels: Sequence[int],  # 0: drill, 1: comp, 2: jump, 3: sum
        reliability_floor: float = 0.75,
    ) -> Tuple[float, float]:
        """Train regularized multinomial logistic head + calibrate temperature & gamma_conf (App. B.3)."""
        if len(queries) == 0:
            return 0.0, self.gamma_conf
        x = np.stack([self.extract_discourse_features(q, t) for q, t in zip(queries, turn_indices)], axis=0)
        y = np.asarray(labels, dtype=np.int64)

        from scipy.optimize import minimize

        n, d = x.shape
        k_cls = 4

        def loss_fn(params: np.ndarray) -> float:
            w = params[: d * k_cls].reshape(d, k_cls)
            b = params[d * k_cls :]
            logits = x @ w + b
            logits -= logits.max(axis=1, keepdims=True)
            log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
            nll = -np.mean(log_probs[np.arange(n), y])
            reg = 0.01 * np.sum(w * w)
            return float(nll + reg)

        init = np.zeros(d * k_cls + k_cls, dtype=np.float64)
        # Warm-start diagonal on cue features
        init_w = init[: d * k_cls].reshape(d, k_cls)
        for c_i in range(4):
            init_w[c_i, c_i] = 2.0
        res = minimize(loss_fn, init, method="BFGS", options={"maxiter": 150})
        self.weights = res.x[: d * k_cls].reshape(d, k_cls)
        self.bias = res.x[d * k_cls :]

        # Temperature scaling on dev logits to minimize NLL & compute ECE
        raw_logits = x @ self.weights + self.bias
        best_t = 1.0
        best_nll = float("inf")
        for cand_t in np.linspace(0.25, 2.5, 46):
            scaled = raw_logits / cand_t
            scaled -= scaled.max(axis=1, keepdims=True)
            lp = scaled - np.log(np.exp(scaled).sum(axis=1, keepdims=True))
            nll = float(-np.mean(lp[np.arange(n), y]))
            if nll < best_nll:
                best_nll = nll
                best_t = float(cand_t)
        self.temperature = best_t

        probs = np.exp((raw_logits / self.temperature) - (raw_logits / self.temperature).max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        confs = probs.max(axis=1)
        preds = probs.argmax(axis=1)
        correct = (preds == y).astype(np.float64)

        # Expected Calibration Error (10 bins)
        ece = 0.0
        bins = np.linspace(0.0, 1.0, 11)
        for b_idx in range(10):
            mask = (confs >= bins[b_idx]) & (confs < bins[b_idx + 1] if b_idx < 9 else confs <= 1.0)
            if np.any(mask):
                ece += (np.sum(mask) / n) * abs(float(np.mean(correct[mask]) - np.mean(confs[mask])))

        # Choose gamma_conf cutoff below which accuracy drops below reliability_floor
        chosen_gamma = 0.70
        for cutoff in np.linspace(0.45, 0.90, 19):
            high_mask = confs >= cutoff
            if np.any(high_mask) and float(np.mean(correct[high_mask])) >= reliability_floor:
                chosen_gamma = float(cutoff)
                break
        self.gamma_conf = chosen_gamma
        return float(ece), self.gamma_conf

    def predict(
        self,
        query: str,
        turn_idx: int = 0,
        candidate_entities: Optional[Iterable[str]] = None,
    ) -> IntentPrediction:
        """Predict (u^(r) in Delta^3, c_conf in [0,1], delta_rel in {-1, 0, +1}) (Sec. 4.1)."""
        q_low = query.lower().strip()
        feats = self.extract_discourse_features(query, turn_idx)

        if self.weights is not None and self.bias is not None:
            logits = feats @ self.weights + self.bias
        else:
            # Calibrated discourse scoring prior to/without dev fit
            logits = np.array(
                [
                    1.6 * feats[0] + 0.8 * feats[4] + 1.1 * (feats[6] + feats[7]) + 0.9 * feats[9],
                    2.1 * feats[1],
                    1.9 * feats[2] + 0.3 * (1.0 - feats[4]) * (1.0 - feats[5]),
                    2.1 * feats[3] + 0.7 * feats[5] * (1.0 - max(feats[0], feats[1], feats[9])),
                ],
                dtype=np.float64,
            )
            if np.max(logits) < 0.25:
                # Ambiguous short follow-up -> low-confidence distribution
                logits = np.array([0.55, 0.20, 0.15, 0.10], dtype=np.float64)

        probs = _softmax(logits, temperature=self.temperature)
        c_conf = float(np.max(probs))

        # Relative temporal coreference shift \delta_rel in {-1, 0, +1}
        if turn_idx > 0 and any(cue in q_low for cue in _FORWARD_CUES):
            delta_rel = 1
        elif turn_idx > 0 and any(cue in q_low for cue in _BACKWARD_CUES):
            delta_rel = -1
        else:
            delta_rel = 0

        # Detect queried entities from video entity vocabulary or capitalized spans
        pool = list(candidate_entities) if candidate_entities is not None else self.known_entities
        matched_ents = tuple(sorted({ent for ent in pool if ent.lower() in q_low}))

        # Audio relevance confidence AudioConf(q^(r)) in [0, 1] (Eq. 7)
        has_audio_cue = any(cue in q_low for cue in _AUDIO_QUERY_CUES)
        has_pure_ocr_cue = any(cue in q_low for cue in ("formula", "equation", "whiteboard", "chart", "small text"))
        if has_audio_cue and not has_pure_ocr_cue:
            audio_conf = 0.95
        elif has_pure_ocr_cue and not has_audio_cue:
            audio_conf = 0.35
        else:
            audio_conf = 0.75

        return IntentPrediction(
            u_drill=float(probs[0]),
            u_comp=float(probs[1]),
            u_jump=float(probs[2]),
            u_sum=float(probs[3]),
            c_conf=c_conf,
            delta_rel=delta_rel,
            queried_entities=matched_ents,
            audio_conf=audio_conf,
        )
