# TurnFocus Codebase Audit & Manuscript Traceability

This document maps sections, equations, and algorithms of the companion TurnFocus manuscript (paper URL and arXiv/DOI will be populated upon camera-ready publication) to exact files in this repository. For double-blind review submissions, generate a strictly anonymized supplementary archive with `bash scripts/export_anonymous_snapshot.sh` rather than linking this public repository.

> **Note**: This repository has not yet been externally evaluated by an Artifact Evaluation Committee. The archived GPU results under `results/synthetic_gpu_sanity_check/` are from a controlled synthetic sanity-check suite exercising real unmodified HuggingFace model weights (`Qwen/Qwen3-VL-8B-Instruct`, `Qwen/Qwen3-VL-2B-Instruct`, `openai/clip-vit-base-patch32`), **not** full 8-benchmark real-video runs.

---

## 1. Repository Directory Tree

```text
.
├── .gitignore                             # Ignores Python caches, build artifacts, and OS metadata
├── README.md                              # Quickstart, synthetic sanity-check scope, and integrity checks
├── ARTIFACT_APPENDIX.md                   # Self-assessed reproducibility appendix (no external badge claimed)
├── TREE_AUDIT.md                          # Equation-to-code traceability matrix
├── LICENSE                                # MIT License (code)
├── CITATION.cff                           # Citation metadata (updated with full authors/DOI upon publication)
├── zenodo.json                            # Draft archival metadata for future Zenodo deposit
├── Dockerfile                             # Portable CUDA 12.4 container spec (see provenance.json for Colab runtime)
├── Makefile                               # Unified entrypoints (install-cpu, test, smoke, dry-run, verify-archive, anon-zip)
├── pyproject.toml                         # PEP 621 installable package (`turnfocus`)
├── requirements-cpu.txt                   # Minimal pinned CPU-only dependencies (tests, smoke, --dry-run)
├── requirements-colab.txt                 # GPU/HuggingFace dependencies for Colab/CUDA runs
├── SHA256SUMS.txt                         # Cryptographic SHA-256 manifest of tracked repository files
│
├── configs/
│   ├── default_hyperparams.yaml           # Loaded by TurnFocusHyperparams.from_yaml() & load_operating_profiles_from_yaml()
│   ├── colab_ae_suite.yaml                # Loaded by scripts/run_colab_ae.py --config
│   └── full_cluster_benchmark.yaml        # Reference cluster launch specification template (external Slurm/Ray)
│
├── src/turnfocus/
│   ├── __init__.py
│   ├── types.py                           # Core typed dataclasses (AtomicClip, ContiguousRun, OperatingProfile, etc.)
│   ├── indexing/                          # Sec. 3.1-3.2: segmentation.py, dig.py, offline_index.py
│   ├── belief/                            # Sec. 4.1: router.py, banded_filter.py, uncertainty_probe.py
│   ├── candidates/                        # Sec. 4.2: nms_stratified.py
│   ├── allocation/                        # Sec. 4.3-4.4: objective.py, lazy_greedy.py, lp_audit.py
│   ├── baselines/                         # Sec. 5.3: allocators.py (5 baselines)
│   ├── models/                            # Sec. 3.3 & 5.1: prompt_templates.py, mllm_runner.py
│   ├── datasets/                          # Sec. 5.2: split_guard.py, hf_benchmarks.py
│   ├── calibration/                       # App. B.3: dev_calibrator.py
│   ├── evaluation/                        # Sec. 5.4 & 6.4: metrics.py, pipeline.py, ablations_and_sweeps.py, cli.py
│   └── repro/                             # Provenance & LaTeX exporter: provenance.py, latex_exporter.py
│
├── tests/
│   ├── __init__.py
│   ├── test_submodularity_and_lp.py       # Lemma 1, Theorem 1(a) (norm/mono/submod), Theorem 1(b) (OPT <= UB(S))
│   ├── test_belief_and_candidates.py      # Column-stochastic T_band, shrinkage, clamped eta_mod, exact M_top+M_strat, 1D NMS
│   └── test_split_guard_and_prompts.py    # Video-disjoint split guard, 2048-tok history cap, Eq. 17 pos budget, prompts
│
├── scripts/
│   ├── run_smoke_test.sh                  # <60s local mathematical & functional verification
│   ├── run_colab_ae.py                    # Controlled synthetic GPU sanity-check runner (supports --dry-run, --max-samples, --seeds)
│   ├── verify_archive.py                  # Automated SHA-256, JSON schema, and archive-vs-candidate diff verifier
│   ├── launch_colab_cli.sh                # Google Colab CLI helper script
│   ├── export_anonymous_snapshot.sh       # Builds a metadata-free zip for double-blind review submission
│   └── generate_checksums.py              # SHA-256 manifest generator (excludes OS metadata)
│
└── results/
    └── synthetic_gpu_sanity_check/        # Archived A100-40GB synthetic sanity-check outputs & provenance
```

---

## 2. Equation-to-Code Traceability Matrix

| Manuscript Reference | Mathematical / Algorithmic Component | Implementation File | Automated Property Test |
| :--- | :--- | :--- | :--- |
| **Sec. 3.1, Eq. (1)** | Contiguous runs $\mathcal{R}$ ($|g| \le L_{\mathrm{run}}$), coarse buckets $\mathcal{T}$ | `src/turnfocus/indexing/segmentation.py` | `tests/test_belief_and_candidates.py` |
| **Sec. 3.2, Eq. (2)** | L0 index & DIG $\phi_{\mathrm{dig}}(c)$ (`Area_ocr`, `Edge_hf`, ECDF `Calib`) | `src/turnfocus/indexing/dig.py`, `offline_index.py` | `tests/test_submodularity_and_lp.py` |
| **Sec. 3.3, Eq. (3)** | Disjoint $(g_j, \tau_j)$, $\tau_{\mathrm{prev}}(c)$ upgrade state, retrieve-then-prefill | `src/turnfocus/types.py`, `models/mllm_runner.py` | `tests/test_split_guard_and_prompts.py` |
| **Sec. 4.1, Eqs. (4)–(7)** | $O(M)$ $\mathbf{T}_{\mathrm{band}}(\delta_{\mathrm{rel}})$, shrinkage $\rho_{\mathrm{shrink}}$, clamped $\eta_{\mathrm{mod}}$, 2B probe | `src/turnfocus/belief/banded_filter.py`, `uncertainty_probe.py` | `tests/test_belief_and_candidates.py` |
| **Sec. 4.2, Eqs. (8)–(10)** | Prefix-sum $b_{\mathrm{run}}(g)$, exact $M_{\mathrm{top}} + M_{\mathrm{strat}} = M_{\mathrm{cand}}$, 1D NMS | `src/turnfocus/candidates/nms_stratified.py` | `tests/test_belief_and_candidates.py` |
| **Sec. 4.3, Eqs. (11)–(17), Alg. 1** | Facility weight $w_{\mathrm{fac}}$, offline $\mu_B$, smooth $\tilde{F}_{\mathbf{u}}$, $B_{\mathrm{pos}}$, lazy greedy | `src/turnfocus/allocation/objective.py`, `lazy_greedy.py` | `tests/test_submodularity_and_lp.py` |
| **Sec. 4.4, Thm. 1, Eq. (18)** | Monotone submodularity & HiGHS LP upper bound $\widetilde{\mathrm{OPT}} \le \mathrm{UB}(S)$ | `src/turnfocus/allocation/lp_audit.py` | `tests/test_submodularity_and_lp.py` |
