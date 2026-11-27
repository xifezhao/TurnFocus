"""Unmodified BF16 MLLM retrieve-then-full-prefill runner & decoupled 2B reranker (Sec. 5.1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np

from turnfocus.models.prompt_templates import (
    format_llm_judge_prompt,
    format_retrieve_then_prefill_prompt,
    parse_judge_verdict,
)
from turnfocus.types import ResolutionTier, RunTierAction, VideoIndex


@dataclass
class SpatialTierSpecs:
    """Pixel dimensions realizing R_L1 (low-res thumbnail) vs. R_L2 (high-res OCR fidelity)."""

    l1_size: Tuple[int, int] = (84, 84)
    l2_size: Tuple[int, int] = (560, 560)


class MLLMBackboneRunner:
    """Interfaces strictly via retrieve-then-full-prefill with unmodified HuggingFace MLLMs (Sec. 3.3 & 5.1).

    Supports `Qwen/Qwen3-VL-8B-Instruct`, `Qwen/Qwen3-VL-30B-A3B-Instruct`,
    and `Qwen/Qwen3-Omni-30B-A3B-Instruct` with greedy decoding (temperature 0.0)
    and standard FlashAttention / SDPA kernels.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        device_map: str = "auto",
        tier_specs: Optional[SpatialTierSpecs] = None,
    ) -> None:
        self.model_id = model_id
        self.device_map = device_map
        self.tier_specs = tier_specs or SpatialTierSpecs()
        self.model: Any = None
        self.processor: Any = None
        self.commit_sha: str = "unloaded"

    def load(self) -> None:
        """Lazy-load unmodified BF16 checkpoint & processor from HuggingFace Hub."""
        if self.model is not None:
            return
        import torch
        from huggingface_hub import model_info
        from transformers import AutoModelForImageTextToText, AutoProcessor

        try:
            info = model_info(self.model_id)
            self.commit_sha = info.sha or "unknown"
        except Exception:
            self.commit_sha = "local_or_cached"

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        load_kwargs: Dict[str, Any] = {
            "dtype": dtype,
            "device_map": self.device_map,
            "trust_remote_code": True,
        }
        # For 8B on >=20GB GPU, force full GPU residency (cuda:0) to avoid slow CPU offload
        if "8B" in self.model_id and torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if vram_gb >= 20.0:
                load_kwargs["device_map"] = {"": 0}
        # For 30B-MoE on 40GB A100 with high-RAM host, allow clean BF16 CPU offload without quantization
        elif "30B" in self.model_id and torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if vram_gb < 70.0:
                load_kwargs["max_memory"] = {0: f"{max(24, int(vram_gb) - 4)}GiB", "cpu": "64GiB"}
                load_kwargs["offload_folder"] = "/tmp/hf_offload"

        self.model = AutoModelForImageTextToText.from_pretrained(self.model_id, **load_kwargs)
        self.model.eval()

    def unload(self) -> None:
        """Release GPU/CPU memory cleanly between multi-backbone evaluations."""
        import gc
        try:
            import torch
        except ImportError:
            torch = None
        self.model = None
        self.processor = None
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate_answer(
        self,
        index: VideoIndex,
        selected_actions: Sequence[RunTierAction],
        clip_frame_fetcher: Callable[[int, Tuple[int, int]], Any],
        history: Sequence[Tuple[str, str]],
        query: str,
        is_multiple_choice: bool = False,
        options: Optional[Sequence[str]] = None,
        max_new_tokens: int = 96,
    ) -> str:
        """Decode selected runs at assigned spatial tiers and run deterministic full prefill."""
        self.load()
        import torch

        prompt_text = format_retrieve_then_prefill_prompt(
            index=index,
            selected_actions=selected_actions,
            history=history,
            current_query=query,
            is_multiple_choice=is_multiple_choice,
            options=options,
        )

        content_blocks: List[Dict[str, Any]] = []
        images_list: List[Any] = []
        for act in selected_actions:
            size_hw = (
                self.tier_specs.l2_size if act.tier == ResolutionTier.R_L2 else self.tier_specs.l1_size
            )
            for c_id in act.run.clip_indices:
                pil_img = clip_frame_fetcher(c_id, size_hw)
                if pil_img is not None:
                    content_blocks.append({"type": "image", "image": pil_img})
                    images_list.append(pil_img)

        content_blocks.append({"type": "text", "text": prompt_text})
        messages = [{"role": "user", "content": content_blocks}]

        rendered = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        proc_kwargs: Dict[str, Any] = {"text": [rendered], "padding": True, "return_tensors": "pt"}
        if images_list:
            proc_kwargs["images"] = images_list

        inputs = self.processor(**proc_kwargs)
        inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

        with torch.inference_mode():
            out_ids = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )
        in_len = inputs["input_ids"].shape[1]
        gen_tokens = out_ids[:, in_len:]
        decoded = self.processor.batch_decode(gen_tokens, skip_special_tokens=True)[0]
        return decoded.strip()

    def grade_open_ended_with_rubric(
        self,
        question: str,
        reference_answer: str,
        candidate_answer: str,
    ) -> Tuple[int, str]:
        """Deterministic temperature-0.0 binary judge evaluation (Fig. 5)."""
        self.load()
        import torch

        rubric_prompt = format_llm_judge_prompt(
            question=question,
            reference_answer=reference_answer,
            candidate_answer=candidate_answer,
        )
        messages = [{"role": "user", "content": [{"type": "text", "text": rubric_prompt}]}]
        rendered = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[rendered], return_tensors="pt")
        inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.inference_mode():
            out_ids = self.model.generate(**inputs, do_sample=False, max_new_tokens=80)
        gen = out_ids[:, inputs["input_ids"].shape[1] :]
        text = self.processor.batch_decode(gen, skip_special_tokens=True)[0]
        return parse_judge_verdict(text)


class Decoupled2BReranker:
    """Optional decoupled 2B auxiliary MLLM mid-layer (L11-L15) cross-attention probe (Sec. 4.1)."""

    def __init__(self, model_id: str = "Qwen/Qwen3-VL-2B-Instruct", device: str = "cuda") -> None:
        self.model_id = model_id
        self.device = device
        self.model: Any = None
        self.processor: Any = None
        self.commit_sha: str = "unloaded"

    def load(self) -> None:
        if self.model is not None:
            return
        import torch
        from huggingface_hub import model_info
        from transformers import AutoModelForImageTextToText, AutoProcessor

        try:
            self.commit_sha = model_info(self.model_id).sha or "unknown"
        except Exception:
            self.commit_sha = "local_or_cached"

        dtype = torch.bfloat16 if (self.device == "cuda" and torch.cuda.is_available()) else torch.float32
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        # Load auxiliary 2B model with eager attention ONLY on this decoupled model so main backbone keeps FlashAttention!
        dmap = {"": 0} if (self.device == "cuda" and torch.cuda.is_available()) else {"": "cpu"}
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            dtype=dtype,
            attn_implementation="eager",
            device_map=dmap,
            trust_remote_code=True,
        )
        self.model.eval()

    def score_clips(
        self,
        query: str,
        clip_indices: np.ndarray,
        index: VideoIndex,
        clip_frame_fetcher: Callable[[int, Tuple[int, int]], Any],
    ) -> np.ndarray:
        """Single-pass mid-layer (L11-L15) attention / relevance scoring over top-K clips."""
        self.load()
        import torch

        scores = np.zeros(len(clip_indices), dtype=np.float64)
        for idx_pos, c_id in enumerate(clip_indices):
            c_int = int(c_id)
            img = clip_frame_fetcher(c_int, (224, 224))
            asr = index.clips[c_int].asr_text
            prompt = f"Clip transcript: {asr}\nQuery: {query}\nIs this clip relevant? Yes or No:"
            content: List[Dict[str, Any]] = []
            imgs = []
            if img is not None:
                content.append({"type": "image", "image": img})
                imgs.append(img)
            content.append({"type": "text", "text": prompt})
            rendered = self.processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )
            kwargs: Dict[str, Any] = {"text": [rendered], "return_tensors": "pt"}
            if imgs:
                kwargs["images"] = imgs
            inputs = self.processor(**kwargs)
            inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = self.model(**inputs, output_attentions=True, return_dict=True)
                attns = getattr(outputs, "attentions", None)
                if attns is not None and len(attns) >= 12:
                    lo = min(11, len(attns) - 1)
                    hi = min(16, len(attns))
                    # Mean attention mass from final query token onto prefix multimodal context across L11-L15
                    mid_vals = [
                        float(attns[layer_i][0, :, -1, :-4].mean().detach().cpu().item())
                        for layer_i in range(lo, hi)
                    ]
                    attn_score = float(np.mean(mid_vals)) if mid_vals else 0.0
                else:
                    attn_score = 0.0

                # Combine mid-layer attention concentration with 'Yes' vs 'No' next-token log-odds
                logits = outputs.logits[0, -1, :].float().detach().cpu()
                yes_id = self.processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
                no_id = self.processor.tokenizer.encode("No", add_special_tokens=False)[0]
                log_odds = float((logits[yes_id] - logits[no_id]).item())
                scores[idx_pos] = log_odds + 2.0 * attn_score

        return scores
