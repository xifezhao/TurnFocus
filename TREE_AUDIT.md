# TurnFocus Complete Experimental Codebase Audit & Tree Specification

This document provides a strict 1-to-1 audit between the ACL manuscript (*TurnFocus: Intent-Adaptive Submodular Evidence Grounding for Multi-Turn Long-Video Question Answering*, `../manuscript.tex`), IEEE / ACM Four-Badge Artifact Evaluation requirements (*Artifacts Available*, *Artifacts Evaluated — Functional*, *Artifacts Evaluated — Reusable*, *Results Reproduced*), and every file in this repository.

---

## 1. Audited Directory Tree (`artifact/`)

```text
artifact/
├── README.md                              # Quickstart (<5 min smoke test), full & Colab CLI reproduction guide
├── ARTIFACT_APPENDIX.md                   # Formal IEEE/ACM 4-Badge & ACL ARR Reproducibility Appendix
├── TREE_AUDIT.md                          # 1-to-1 audit mapping manuscript equations/tables/figures to source files
├── LICENSE                                # MIT License (code); upstream dataset licenses documented separately
├── CITATION.cff                           # Machine-readable citation metadata (Artifacts Available badge)
├── zenodo.json                            # Archival DOI / Zenodo deposit metadata (Artifacts Available badge)
├── Dockerfile                             # Reproducible CUDA 12.4 + Python 3.11 + PyTorch container spec
├── pyproject.toml                         # PEP 621 installable package (`turnfocus`), dependencies & CLI scripts
├── requirements-colab.txt                 # Pinned runtime dependencies for Google Colab GPU VMs
│
├── configs/
│   ├── default_hyperparams.yaml           # Exact Table 4 (tab:app_hyperparams) initialization & sweep grids
│   ├── colab_ae_suite.yaml                # Stratified real-model/real-data Google Colab GPU execution config (>=5 seeds)
│   └── full_cluster_benchmark.yaml        # Full multi-TB 8-benchmark sharded execution config
│
├── src/
│   └── turnfocus/
│       ├── __init__.py                    # Public API exports & semantic version
│       ├── types.py                       # Immutable/typed dataclasses: AtomicClip, ContiguousRun, VideoIndex, Profile, etc.
│       │
│       ├── indexing/
│       │   ├── __init__.py
│       │   ├── segmentation.py            # Sec. 3.1: Shot/speech clip segmentation, coarse buckets I_w, runs R (Eq. 1)
│       │   ├── dig.py                     # Sec. 3.2: Dense Information Guard phi_dig(c), OCR area + Sobel/Laplacian + ECDF Calib (Eq. 2)
│       │   └── offline_index.py           # Sec. 3.2: Non-destructive L0 index (Whisper ASR, ASR_Density, SigLIP max-pool KFs, top-K_nbr, soft m_a(g))
│       │
│       ├── belief/
│       │   ├── __init__.py
│       │   ├── router.py                  # Sec. 4.1 & App. C.3 (Table 8): Intent u^(r), confidence c_conf, coref shift delta_rel + temp scaling
│       │   ├── banded_filter.py           # Sec. 4.1: O(M) column-stochastic T_band(delta_rel), shrinkage rho_shrink, clamped eta_mod (Eqs. 4-7)
│       │   └── uncertainty_probe.py       # Sec. 4.1: Top-K Shannon entropy H_ret > tau_H & decoupled 2B MLLM mid-layer (L11-L15) reranker
│       │
│       ├── candidates/
│       │   ├── __init__.py
│       │   └── nms_stratified.py          # Sec. 4.2: Prefix-sum b_run(g), exact split M_top + M_strat = M_cand, 1D IoU NMS & bucket stratification (Eqs. 8-10)
│       │
│       ├── allocation/
│       │   ├── __init__.py
│       │   ├── objective.py               # Sec. 4.3: Facility weight w_fac (Eq. 11), F_drill, tilde_F_sum, h_a, F_comp, tilde_F_u (Eqs. 12-16)
│       │   ├── lazy_greedy.py             # Sec. 4.3 & Alg. 1: Incremental-state O(U^2 K_nbr) max-heap lazy greedy w/ matroid, resource & pos budgets (Eq. 17)
│       │   └── lp_audit.py                # Sec. 4.4 & Thm. 1(b): Off-path HiGHS LP upper bound UB(S) (Eq. 18) + exact MILP oracle for verification
│       │
│       ├── baselines/
│       │   ├── __init__.py
│       │   └── allocators.py              # Sec. 5.3: (i) Uniform FPS, (ii) Top-k CLIP/ASR, (iii) Static Token Pruning, (iv) Pure FL, (v) Offline Agentic Search
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── prompt_templates.py        # App. C.4: Verbatim retrieve-then-full-prefill prompt (Fig. 4) & temp-0.0 binary judge prompt (Fig. 5)
│       │   └── mllm_runner.py             # Sec. 3.3 & 5.1: Unmodified BF16 FlashAttention runner for Qwen3-VL-8B, Qwen3-VL-30B-A3B, Qwen3-Omni-30B-A3B
│       │
│       ├── datasets/
│       │   ├── __init__.py
│       │   ├── split_guard.py             # Sec. 5.2 & App. C.2: Strict video-disjoint partition verifier, 4-stratum tagger, 2048-tok history compressor
│       │   └── hf_benchmarks.py           # Sec. 5.2 & Table 7: Loaders for Video-MME, LongVideoBench, MLVU, LVBench, CG-Bench, SVBench, OmniMMI, VideoMemBench
│       │
│       ├── calibration/
│       │   ├── __init__.py
│       │   └── dev_calibrator.py          # App. B.3: Frozen dev-only calibration of mu_B, DIG alpha_ocr/ECDF/tau_dig, router ECE/gamma_conf, and tau_H (<15% rate)
│       │
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── metrics.py                 # Sec. 5.4: MC & LLM-Judge Acc, Cohen's kappa, Temporal IoU, DIG Retention, Pareto, LP percentiles, P50/P95 latency, 95% CI
│       │   ├── ablations_and_sweeps.py    # Sec. 6.4 (Table 3: i-vii), App. B.4 (Table 6: L_run, M_cand, K_nbr, theta_nms), App. D.2 (Table 11: Silent breakdown)
│       │   └── pipeline.py                # End-to-end multi-seed session orchestrator separating indexing, online front-end, prefill, and off-path LP audit
│       │
│       └── repro/
│           ├── __init__.py
│           ├── provenance.py              # Table 5 (tab:app_repro): Deterministic seeding, HF revision SHA pinning, GPU/CUDA/PyTorch & GPU-hour logger
│           └── latex_exporter.py          # Automated generator for Tables 1, 2, 3, 5, 6, 9, 10, 11 and Figure 3 pgfplots coordinates
│
├── tests/
│   ├── test_submodularity_and_lp.py       # Exhaustive property tests for Lemma 1, Theorem 1(a) (norm/mono/submod), and Theorem 1(b) (MILP OPT <= LP UB(S))
│   ├── test_belief_and_candidates.py      # Tests O(M) column-stochastic T_band, shrinkage, clamped eta_mod in [0,1], M_top+M_strat==M_cand, 1D NMS
│   └── test_split_guard_and_prompts.py    # Tests zero video-ID leakage across splits, Eq. 17 positional budget exactness, and verbatim prompt formatting
│
└── scripts/
    ├── run_smoke_test.sh                  # < 60s functional & mathematical verification script (Functional Badge)
    ├── run_colab_ae.py                    # Self-contained Google Colab GPU experiment driver (real HF data + real Qwen3 MLLM + all tables/sweeps/LP audit)
    ├── launch_colab_cli.sh                # Automated host-side orchestration using Google `colab` CLI (`new`, `upload`, `install`, `exec`, `download`, `log`, `stop`)
    └── generate_checksums.py              # SHA-256 artifact integrity manifest generator
```

---

## 2. Manuscript-to-Code Traceability Matrix

| Manuscript Reference | Mathematical / Experimental Requirement | Exact Implementation Module | Automated Test / Audit |
| :--- | :--- | :--- | :--- |
| **Sec. 3.1, Eq. (1)** | Contiguous runs $\mathcal{R}$ with $|g| \le L_{\mathrm{run}}$, coarse buckets $\mathcal{T}$ | `turnfocus/indexing/segmentation.py` | `tests/test_belief_and_candidates.py` |
| **Sec. 3.2, Eq. (2)** | L0 index (`ASR_Density`, max-pool KFs, $\mathcal{N}_{K_{\mathrm{nbr}}}$, soft $m_a(g)$) & DIG $\phi_{\mathrm{dig}}(c)$ | `turnfocus/indexing/dig.py`, `offline_index.py` | `tests/test_submodularity_and_lp.py` |
| **Sec. 3.3, Eq. (3)** | Disjoint $(g_j, \tau_j)$, $\tau_{\mathrm{prev}}(c)$ upgrade state, retrieve-then-full-prefill | `turnfocus/types.py`, `models/mllm_runner.py` | `tests/test_split_guard_and_prompts.py` |
| **Sec. 4.1, Eqs. (4)–(7)** | $O(M)$ $\mathbf{T}_{\mathrm{band}}(\delta_{\mathrm{rel}})$, shrinkage $\rho_{\mathrm{shrink}}$, clamped $\eta_{\mathrm{mod}}$, 2B probe $H_{\mathrm{ret}} > \tau_H$ | `turnfocus/belief/banded_filter.py`, `uncertainty_probe.py` | `tests/test_belief_and_candidates.py` |
| **Sec. 4.2, Eqs. (8)–(10)** | Prefix-sum $b_{\mathrm{run}}(g)$, exact $M_{\mathrm{top}} + M_{\mathrm{strat}} = M_{\mathrm{cand}}$, 1D NMS & bucket stratification | `turnfocus/candidates/nms_stratified.py` | `tests/test_belief_and_candidates.py` |
| **Sec. 4.3, Eqs. (11)–(17), Alg. 1** | Facility upgrade weight $w_{\mathrm{fac}}$, offline $\mu_B$, smooth $\tilde{F}_{\mathbf{u}}$, pos budget $B_{\mathrm{pos}}$, lazy greedy | `turnfocus/allocation/objective.py`, `lazy_greedy.py` | `tests/test_submodularity_and_lp.py` |
| **Sec. 4.4, Thm. 1, Eq. (18), App. A** | Monotone submodularity & data-dependent HiGHS LP upper bound $\widetilde{\mathrm{OPT}} \le \mathrm{UB}(S)$ | `turnfocus/allocation/lp_audit.py` | `tests/test_submodularity_and_lp.py` |
| **Sec. 5.1–5.4, App. C** | 3 Qwen3 backbones, 8 HF benchmarks, video-disjoint splits, 5 baselines, Fig. 4–5 prompts, Cohen's $\kappa$ | `turnfocus/models/`, `datasets/`, `baselines/`, `evaluation/` | `tests/test_split_guard_and_prompts.py` |
| **Sec. 6.1–6.4, App. B & D** | Main Tables 1 & 9, Pareto/DIG Fig. 3, LP/Latency Table 2, Ablations Table 3, Sweeps Table 6, Silent Table 11 | `turnfocus/evaluation/ablations_and_sweeps.py`, `repro/latex_exporter.py` | `scripts/run_colab_ae.py` |

---

## 3. IEEE / ACM Four-Badge Compliance Checklist

1. **Artifacts Available**: Permanent archival structure with `CITATION.cff`, `zenodo.json`, `LICENSE`, `Dockerfile`, and SHA-256 checksum manifest (`scripts/generate_checksums.py`).
2. **Artifacts Evaluated — Functional**: Complete documentation (`README.md`, `ARTIFACT_APPENDIX.md`), zero-fabrication real algorithmic implementations, `<60s` smoke & property verification suite (`pytest`), and exact mathematical invariant checks.
3. **Artifacts Evaluated — Reusable**: Clean PEP-621 typed Python package (`src/turnfocus`), modular interfaces for adding new MLLM backbones or datasets, schema-driven YAML configs (`configs/`), and CLI drivers.
4. **Results Reproduced**: End-to-end automated execution on Google Colab GPU hardware via `colab` CLI (`scripts/launch_colab_cli.sh` -> `scripts/run_colab_ae.py`), $\ge 5$ random seeds with 95% confidence intervals, cryptographic provenance logging (`provenance.json`), replayable `.ipynb` session export, and automated LaTeX table/plot coordinate export.
