# Self-Assessed Reproducibility & Artifact Appendix

> **Disclaimer**: This appendix is a **self-assessed reproducibility checklist** prepared in the standard ACM/IEEE Artifact Evaluation and ACL ARR format to assist reviewers. No external Artifact Evaluation Committee has reviewed this artifact yet, and **no ACM/IEEE badges have been awarded**.

## A.1 Abstract
This artifact provides the reference source code, configuration files, mathematical property test suite, HuggingFace MLLM/CLIP runners, HiGHS linear-programming optimality auditor, and GPU sanity-check scripts for **TurnFocus: Intent-Adaptive Submodular Evidence Grounding for Multi-Turn Long-Video Question Answering**.

---

## A.2 Artifact Meta-Information Checklist
- **Algorithm**: TurnFocus ($O(M)$ banded-kernel belief filter, exact-cardinality 1D temporal NMS + bucket stratification, smooth intent-weighted submodular lazy-greedy allocator, and off-path HiGHS LP optimality auditor).
- **Program**: Python `>= 3.10` (`turnfocus` PEP-621 package).
- **Solvers & Libraries**: NumPy, SciPy HiGHS (`scipy.optimize.linprog`), PyTorch `>= 2.4`, HuggingFace `transformers >= 4.51`.
- **Models Exercised in Sanity Check (Unmodified Open-Weight Checkpoints)**:
  - Dense Vision-Language Backbone: `Qwen/Qwen3-VL-8B-Instruct` (BF16, pinned SHA `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`)
  - Auxiliary Decoupled 2B Probe: `Qwen/Qwen3-VL-2B-Instruct` (pinned SHA `89644892e4d85e24eaac8bacfd4f463576704203`)
  - Offline Keyframe Dual-Encoder: `openai/clip-vit-base-patch32` (pinned SHA `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`)
  - 30B-MoE Backbones (`Qwen/Qwen3-VL-30B-A3B-Instruct`, `Qwen/Qwen3-Omni-30B-A3B-Instruct`): Supported by `MLLMBackboneRunner`; require 80GB+ GPU hardware and are **not** included in the archived 40GB sanity-check run.
- **Data Scope & Transparency**:
  - The archived run in `results/synthetic_gpu_sanity_check/` uses **controlled synthetic audiovisual episodes** (4 Dev / 15 Test video-disjoint episodes) for end-to-end functional, latency, and LP-audit verification.
  - Full-scale evaluation on the 8 public long-video benchmarks (Video-MME, LongVideoBench, MLVU, LVBench, CG-Bench, SVBench, OmniMMI, VideoMemBench) requires downloading the official benchmark video files under their respective licenses; the included `hf_benchmarks.py` module records official HuggingFace Hub repository IDs and live commit SHAs.
- **Run-time Environment**: Linux/macOS/Windows (CPU unit tests) or NVIDIA CUDA GPU (sanity-check & full evaluation).
- **Output**: Structured JSON (`provenance.json`, `full_empirical_results.json`), Markdown report (`summary_report.md`), and `pgfplots` coordinate snippets (`fig3_pgfplots_coords.tex`).

---

## A.3 Self-Assessed Reproducibility Checklist

1. **Availability & Double-Blind Preparation**: Includes `LICENSE` (MIT), `CITATION.cff` (tagged at `v1.0.0`), `zenodo.json`, `Dockerfile`, `Makefile`, `requirements-cpu.txt`, `SHA256SUMS.txt` (verified with zero missing entries), and `scripts/export_anonymous_snapshot.sh` to package a metadata-free supplementary archive for double-blind review. Full authors, affiliations, paper URL, and Zenodo DOI will be populated upon camera-ready publication.
2. **Functional Verification (Low-Cost CPU + Full GPU)**:
   - Exact implementations of Equations (1)–(18) and Algorithm 1 in `src/turnfocus/`.
   - `make smoke` (`scripts/run_smoke_test.sh`) executes 13 unit tests verifying normalization, monotonicity, submodularity, incremental-vs-exact marginal equivalence, exact discrete optimum $\widetilde{\mathrm{OPT}} \le \mathrm{UB}(S)$, column-stochastic $\mathbf{T}_{\mathrm{band}}$, clamped $\eta_{\mathrm{mod}}$, exact cardinality $M_{\mathrm{top}} + M_{\mathrm{strat}} = M_{\mathrm{cand}}$, YAML config loading, and strict video-disjoint split guards.
   - `make dry-run` executes the full `scripts/run_colab_ae.py` pipeline locally on CPU with `--dry-run --max-samples 6 --seeds 101,202` (zero 20GB checkpoint downloads) and diffs the output structure against the archived JSON via `scripts/verify_archive.py`.
3. **Reusability**:
   - Modular PEP-621 package (`src/turnfocus/`) with typed dataclasses (`types.py`), decoupled `MLLMBackboneRunner` supporting unmodified HuggingFace VLMs via retrieve-then-full-prefill, and YAML configs (`configs/`).
4. **Transparent Provenance**:
   - `results/synthetic_gpu_sanity_check/provenance.json` logs exact hardware, software stack, pinned HuggingFace commit SHAs, and GPU-hours, explicitly distinguished from full multi-benchmark evaluation.
