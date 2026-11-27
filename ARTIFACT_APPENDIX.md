# Artifact Appendix (IEEE / ACM Four-Badge & ACL ARR Specification)

## A.1 Abstract
This artifact provides the complete source code, configuration files, mathematical property test suite, HuggingFace benchmark & MLLM adapters, linear-programming optimality auditor, and automated Google Colab CLI orchestration scripts for **TurnFocus: Intent-Adaptive Submodular Evidence Grounding for Multi-Turn Long-Video Question Answering**. It is structured to satisfy all four IEEE / ACM Artifact Review and Badging v1.1 criteria: **Artifacts Available**, **Artifacts Evaluated — Functional**, **Artifacts Evaluated — Reusable**, and **Results Reproduced**.

---

## A.2 Artifact Meta-Information Checklist
- **Algorithm**: TurnFocus ($O(M)$ banded-kernel belief filter, exact-cardinality 1D temporal NMS + bucket stratification, smooth intent-weighted submodular lazy-greedy allocator, and off-path HiGHS LP optimality auditor).
- **Program**: Python `>= 3.10` (`turnfocus` PEP-621 package).
- **Compilation / Solvers**: NumPy, SciPy HiGHS (`scipy.optimize.linprog`), PyTorch `>= 2.4` with SDPA / FlashAttention, HuggingFace `transformers >= 4.51`.
- **Models (Unmodified Open-Weight Checkpoints)**:
  - Dense Vision-Language Backbone: `Qwen/Qwen3-VL-8B-Instruct` (BF16)
  - Sparse MoE Backbones (80GB GPU): `Qwen/Qwen3-VL-30B-A3B-Instruct`, `Qwen/Qwen3-Omni-30B-A3B-Instruct`
  - Auxiliary Decoupled 2B Probe: `Qwen/Qwen3-VL-2B-Instruct` (mid-layer $L_{11}\text{--}L_{15}$)
  - Offline Keyframe Dual-Encoder: `openai/clip-vit-base-patch32`
- **Data-Set Registry**: Video-MME (`lmms-lab/Video-MME`), LongVideoBench (`longvideobench/LongVideoBench`), MLVU (`MLVU/MVLU`), LVBench (`THUDM/LVBench`), CG-Bench (`CG-Bench/CG-Bench`), SVBench (`yzy666/SVBench`), OmniMMI (`bigai-nlco/OmniMMI`), and VideoMemBench (`VideoMemBench/data`).
- **Run-time Environment**: Ubuntu 22.04+ / macOS (CPU verification) or NVIDIA L4 / A100 / H100 GPU (Google Colab CLI or Docker `Dockerfile`).
- **Metrics**: QA Accuracy (Exact MC & temp-0.0 binary LLM-Judge + Cohen's $\kappa$), Temporal Grounding IoU, DIG Information-Cliff Retention, Accuracy-Token Pareto Curves ($P_0, P_1, P_2$), LP Optimality Ratio ($\tilde{F}_{\mathbf{u}}(S)/\mathrm{UB}(S)$ Median / P10 / Worst), and Wall-Clock Localization Latency (P50 / P95 ms).
- **Output**: Structured JSON (`provenance.json`, `full_empirical_results.json`), Markdown summary tables (`summary_report.md`), drop-in LaTeX `pgfplots` coordinates (`fig3_pgfplots_coords.tex`), and Colab session `.ipynb` logs.
- **Publicly Archivable**: Yes (`LICENSE`, `CITATION.cff`, `zenodo.json`, SHA-256 manifest).

---

## A.3 Compliance with IEEE / ACM Four Badges

1. **Artifacts Available**: Includes `CITATION.cff`, `zenodo.json`, `LICENSE`, `Dockerfile`, and `scripts/generate_checksums.py` producing `SHA256SUMS.txt`.
2. **Artifacts Evaluated — Functional**:
   - Zero-fabrication implementation of Equations (1)–(18) and Algorithm 1.
   - `scripts/run_smoke_test.sh` runs `pytest` verifying normalization, monotonicity, submodularity, incremental-vs-exact equivalence, exact discrete optimum $\widetilde{\mathrm{OPT}} \le \mathrm{UB}(S)$, column-stochastic $\mathbf{T}_{\mathrm{band}}$, clamped $\eta_{\mathrm{mod}}$, exact cardinality $M_{\mathrm{top}} + M_{\mathrm{strat}} = M_{\mathrm{cand}}$, and zero cross-split video leakage.
3. **Artifacts Evaluated — Reusable**:
   - Clean `src/turnfocus/` package layout with typed dataclasses (`turnfocus/types.py`), modular `MLLMBackboneRunner` supporting arbitrary HuggingFace VLM checkpoints via retrieve-then-full-prefill, and YAML experiment configs (`configs/`).
4. **Results Reproduced**:
   - `scripts/launch_colab_cli.sh` provisions a Google Colab GPU VM, installs dependencies, executes `scripts/run_colab_ae.py` across 5 deterministic seeds (`[101, 202, 303, 404, 505]`), logs exact HuggingFace revision SHAs and GPU telemetry to `provenance.json`, downloads all outputs, exports the replayable session notebook, and stops the VM cleanly.

---

## A.4 Extending TurnFocus to Custom Backbones or Benchmarks
- **New MLLM Backbone**: Pass `--backbone <hf_org/model_name>` to `scripts/run_colab_ae.py` or instantiate `MLLMBackboneRunner(model_id=...)` — zero model weight or FlashAttention changes are required.
- **New Video Dataset**: Construct `VideoDialogueEpisode` objects in `turnfocus/datasets/` and pass train/dev/test video IDs through `verify_video_disjoint_splits(...)` prior to `calibrate_on_dev_split(...)`.
