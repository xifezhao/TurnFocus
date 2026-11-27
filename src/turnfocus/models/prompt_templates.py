"""Verbatim prompt templates matching Figure 4 (fig:prompt_eval) and Figure 5 (fig:prompt_judge) of manuscript.tex."""

from __future__ import annotations

import json
import re
from typing import List, Optional, Sequence, Tuple

from turnfocus.types import RunTierAction, VideoIndex


def format_timestamp_mmss(sec: float) -> str:
    """Format seconds as MM:SS (or HH:MM:SS for >= 3600s)."""
    total = max(0, int(round(sec)))
    mm, ss = divmod(total, 60)
    return f"{mm:02d}:{ss:02d}"


def format_retrieve_then_prefill_prompt(
    index: VideoIndex,
    selected_actions: Sequence[RunTierAction],
    history: Sequence[Tuple[str, str]],
    current_query: str,
    is_multiple_choice: bool = False,
    options: Optional[Sequence[str]] = None,
) -> str:
    """Construct exact retrieve-then-full-prefill text prompt matching Figure 4 (App. C.4).

    Identical template shared strictly across TurnFocus and every non-agentic baseline.
    """
    lines: List[str] = [
        "[System] You are a precise multimodal video assistant. Answer the user's question strictly "
        "using the provided timestamped video clips and conversation history. If text or numbers "
        "appear on screen, transcribe them accurately.",
        "",
        "[Video Evidence (Chronological)]",
    ]

    for act in selected_actions:
        g = act.run
        ts_start = format_timestamp_mmss(g.start_sec)
        ts_end = format_timestamp_mmss(g.end_sec)
        asr_snippets = [
            index.clips[c_id].asr_text.strip()
            for c_id in g.clip_indices
            if index.clips[c_id].asr_text.strip()
        ]
        asr_joined = " ".join(asr_snippets) if asr_snippets else "(non-verbal / silent segment)"
        lines.append(f"<{ts_start}--{ts_end}> [Tier={act.tier.label}] Spoken ASR: {asr_joined}")

    lines.append("")
    lines.append("[Prior Conversation History]")
    if history:
        for q_prev, y_prev in history:
            lines.append(f"User: {q_prev}    Assistant: {y_prev}")
    else:
        lines.append("(None — Turn 1)")

    lines.append("")
    lines.append("[Current Turn]")
    q_full = current_query.strip()
    if options:
        opt_block = "\n".join(options)
        q_full = f"{q_full}\nOptions:\n{opt_block}"
    lines.append(f"User: {q_full}")
    if is_multiple_choice:
        lines.append("Respond with the single best option letter.")

    return "\n".join(lines)


def format_llm_judge_prompt(
    question: str,
    reference_answer: str,
    candidate_answer: str,
) -> str:
    """Construct deterministic binary LLM-as-a-judge rubric matching Figure 5 (App. C.4)."""
    return (
        "[System] You are an impartial evaluator grading open-ended multi-turn video QA. "
        "Compare the Candidate Answer against the Reference Answer and Ground-Truth Evidence Span "
        "using temperature 0.0.\n\n"
        "[Rubric]\n"
        "Assign Score = 1 iff the Candidate Answer is factually consistent with the Reference Answer, "
        "directly addresses the question (including exact key numbers/OCR text when required), and "
        "contains no fabricated claims contradicting the reference. Assign Score = 0 otherwise "
        "(incorrect, incomplete on key facts, or hallucinated).\n\n"
        "[Inputs]\n"
        f"Question: {question}\n"
        f"Reference Answer: {reference_answer}\n"
        f"Candidate Answer: {candidate_answer}\n\n"
        "[Output Format (JSON)]\n"
        '{"verdict": 0 or 1, "rationale": "<one concise sentence>"}'
    )


def parse_mc_option_letter(response_text: str, valid_letters: str = "ABCDE") -> str:
    """Deterministically extract option letter from closed-ended MC model output."""
    cleaned = response_text.strip()
    if not cleaned:
        return ""
    if len(cleaned) == 1 and cleaned.upper() in valid_letters:
        return cleaned.upper()
    # 1. Explicit "Answer: B" / "Option (C)" (case-insensitive keyword, standalone option letter)
    m_ans = re.search(
        rf"(?:answer|option|choice)\s*(?:is|:|\-)?\s*\(?([{valid_letters}])\)?(?![A-Za-z])",
        cleaned,
        flags=re.IGNORECASE,
    )
    if m_ans:
        return m_ans.group(1).upper()
    # 2. Parenthesized or dot-suffixed uppercase option: "(B)" or "B." at start or end
    m_paren = re.search(rf"\(([{valid_letters}])\)|(?:^|\s)([{valid_letters}])[\.\):](?:\s|$)", cleaned)
    if m_paren:
        ch = m_paren.group(1) or m_paren.group(2)
        if ch in valid_letters:
            return ch
    # 3. Standalone uppercase letter token (never matches lowercase 'e' in 'The')
    m_tok = re.search(rf"(?<![A-Za-z])([{valid_letters}])(?![A-Za-z])", cleaned)
    if m_tok:
        return m_tok.group(1)
    return ""


def parse_judge_verdict(judge_output: str) -> Tuple[int, str]:
    """Parse JSON verdict from deterministic binary LLM judge output."""
    text = judge_output.strip()
    m = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            verdict = int(obj.get("verdict", 0))
            return (1 if verdict == 1 else 0), str(obj.get("rationale", ""))
        except (ValueError, TypeError, json.JSONDecodeError):
            pass
    if re.search(r'"verdict"\s*:\s*1\b|\bScore\s*=\s*1\b', text):
        return 1, text[:160]
    return 0, text[:160]
