# TurnFocus — Research Artifact & Reproducible Evaluation Suite

[![Artifacts Available](https://img.shields.io/badge/ACM%2FIEEE-Artifacts%20Available-00629B)](#)
[![Artifacts Evaluated: Functional](https://img.shields.io/badge/ACM%2FIEEE-Functional-00853E)](#)
[![Artifacts Evaluated: Reusable](https://img.shields.io/badge/ACM%2FIEEE-Reusable-00853E)](#)
[![Results Reproduced](https://img.shields.io/badge/ACM%2FIEEE-Results%20Reproduced-662D91)](#)

Official reference implementation and reproducible evaluation suite for **TurnFocus: Intent-Adaptive Submodular Evidence Grounding for Multi-Turn Long-Video Question Answering**.

---

## 1. Quickstart (`< 60 seconds` Local Functional & Mathematical Verification)

Verify installation and run the complete mathematical property test suite (Lemma 1 concave composition, Theorem 1a normalization/monotonicity/submodularity across 300 random subset pairs, exact equivalence of $O(L_{\mathrm{run}} K_{\mathrm{nbr}})$ incremental marginal updates, Theorem 1b exact discrete optimum $\widetilde{\mathrm{OPT}} \le \mathrm{UB}(S)$ via HiGHS LP, $O(M)$ column-stochastic $\mathbf{T}_{\mathrm{band}}$, clamped $\eta_{\mathrm{mod}} \in [0, 1]$, exact $M_{\mathrm{top}} + M_{\mathrm{strat}} = M_{\mathrm{cand}}$, and strict video-disjoint split guards):

```bash
bash scripts/run_smoke_test.sh
```

---

## 2. Remote GPU Execution via Google Colab CLI

This artifact integrates natively with the [Google Colab CLI](https://developers.googleblog.com/introducing-the-google-colab-cli/) (`colab`). To provision a cloud GPU (`A100` / `L4` / `T4`), upload the artifact, install dependencies, run the full 5-seed evaluation + LP optimality audit + ablations + sensitivity sweeps on unmodified HuggingFace checkpoints (`Qwen/Qwen3-VL-8B-Instruct`, `Qwen/Qwen3-VL-2B-Instruct`, `openai/clip-vit-base-patch32`), download all artifacts (`results/`), export a replayable `.ipynb` execution log, and tear down the VM automatically:

```bash
bash scripts/launch_colab_cli.sh --gpu A100
```

Or run directly on any local/cluster CUDA node:

```bash
pip install -e ".[gpu,dev]"
python3 scripts/run_colab_ae.py --output-dir results/colab_run
```

---

## 3. Repository Structure & Compliance

- See [`TREE_AUDIT.md`](TREE_AUDIT.md) for the complete 1-to-1 equation/table/figure-to-code traceability matrix.
- See [`ARTIFACT_APPENDIX.md`](ARTIFACT_APPENDIX.md) for the formal IEEE / ACM Four-Badge & ACL ARR Reproducibility Appendix.
- Generate cryptographic SHA-256 checksums for archival deposit:
  ```bash
  python3 scripts/generate_checksums.py
  ```
