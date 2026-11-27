# TurnFocus — Research Artifact & Reproducibility Codebase

> **Review, Anonymity & Badge Status Notice**:
> - **No Official Badges Awarded**: This repository has **not** yet undergone external ACM/IEEE Artifact Evaluation and has **not** been awarded any official artifact badges. [`ARTIFACT_APPENDIX.md`](ARTIFACT_APPENDIX.md) is a **self-assessed reproducibility checklist**.
> - **Double-Blind Review Policy**: This public GitHub repository is the open-source reference codebase and is **not** an anonymized double-blind submission link. During double-blind peer review (e.g., ACL ARR / ACM), authors and reviewers should use a strictly anonymized supplementary archive (generated via `bash scripts/export_anonymous_snapshot.sh`, or hosted on an anonymous mirror such as `anonymous.4open.science`) rather than linking this public repository in the anonymized manuscript.

Reference implementation, mathematical property verification suite, and evaluation scripts for **TurnFocus: Intent-Adaptive Submodular Evidence Grounding for Multi-Turn Long-Video Question Answering**.

---

## 1. Quickstart (`< 60 seconds` Local Mathematical & Functional Verification)

Verify installation and run the complete mathematical property test suite (Lemma 1 concave composition, Theorem 1a normalization/monotonicity/submodularity across 300 random subset pairs, exact equivalence of $O(L_{\mathrm{run}} K_{\mathrm{nbr}})$ incremental marginal updates, Theorem 1b exact discrete optimum $\widetilde{\mathrm{OPT}} \le \mathrm{UB}(S)$ via HiGHS LP, $O(M)$ column-stochastic $\mathbf{T}_{\mathrm{band}}$, clamped $\eta_{\mathrm{mod}} \in [0, 1]$, exact $M_{\mathrm{top}} + M_{\mathrm{strat}} = M_{\mathrm{cand}}$, and strict video-disjoint split guards):

```bash
pip install -e ".[dev]"
bash scripts/run_smoke_test.sh
```

---

## 2. Scope of Included GPU Sanity-Check Results

The archived outputs in [`results/synthetic_gpu_sanity_check/`](results/synthetic_gpu_sanity_check/) come from running **real unmodified BF16 HuggingFace checkpoints** (`Qwen/Qwen3-VL-8B-Instruct`, `Qwen/Qwen3-VL-2B-Instruct`, `openai/clip-vit-base-patch32`) on an `NVIDIA A100-SXM4-40GB` GPU over a **small controlled synthetic audiovisual sanity-check suite** (4 Dev / 15 Test video-disjoint template-constructed episodes).

- **Important**: These synthetic sanity-check runs verify end-to-end model execution, LP optimality auditing, and wall-clock localization latency (`8.98 ms` P50 on $M=120$ clip indices). They are **not** full evaluations on the 8 public real-video benchmarks (Video-MME, LongVideoBench, MLVU, LVBench, CG-Bench, SVBench, OmniMMI, VideoMemBench), which require downloading the official multi-terabyte video corpora separately.

To reproduce the GPU sanity check on a CUDA machine or Google Colab GPU:

```bash
pip install -e ".[gpu,dev]"
python3 scripts/run_colab_ae.py --output-dir results/synthetic_gpu_sanity_check
```

---

## 3. Repository Structure & Integrity Verification

- [`TREE_AUDIT.md`](TREE_AUDIT.md): Mapping from manuscript equations/algorithms to source files.
- [`ARTIFACT_APPENDIX.md`](ARTIFACT_APPENDIX.md): Self-assessed reproducibility appendix following ACM/IEEE AE and ACL ARR guidelines.
- Verify file integrity against [`SHA256SUMS.txt`](SHA256SUMS.txt):
  ```bash
  sha256sum -c SHA256SUMS.txt
  ```
