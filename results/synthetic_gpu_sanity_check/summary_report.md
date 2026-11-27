# TurnFocus Controlled Synthetic GPU Sanity-Check Report

> **Scope Notice (Synthetic Sanity Check — Not Real Public Benchmarks)**: This report logs end-to-end execution of real unmodified HuggingFace model weights (`Qwen/Qwen3-VL-8B-Instruct`, `Qwen/Qwen3-VL-2B-Instruct`, `openai/clip-vit-base-patch32`) on a small controlled synthetic audiovisual testbed (4 Dev / 15 Test video-disjoint template episodes). It is intended for functional, LP-audit, and wall-clock latency verification and **must not** be interpreted as full evaluation on the 8 real public long-video benchmarks.

## 1. Hardware & Pinned Checkpoint Provenance
- **Timestamp (UTC)**: `2026-11-27T02:17:14.342345+00:00`
- **GPU Hardware**: `NVIDIA A100-SXM4-40GB` (39.49 GB VRAM)
- **Stack**: Python `3.13.16`, PyTorch `2.14.1+cu130`, CUDA `13.0`, Transformers `5.24.1`
- **Pinned Checkpoint `Qwen/Qwen3-VL-8B-Instruct`**: `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`
- **Pinned Checkpoint `Qwen/Qwen3-VL-2B-Instruct`**: `89644892e4d85e24eaac8bacfd4f463576704203`
- **Pinned Checkpoint `openai/clip-vit-base-patch32`**: `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`

## 2. Live HuggingFace Benchmark Registry SHAs
- **Video-MME**: `ead1408f75b618502df9a1d8e0950166bf0a2a0b`
- **LongVideoBench**: `60d1c89c1919a198b73be39c2babb213b29d6a5c`
- **MLVU**: `06ddc388aa34746b3abba77972b7a7dfd977f7a3`
- **LVBench**: `0caedb92002cc268bad486449e551c76f0485670`
- **CG-Bench**: `aa63c0b7f74b5b43b5e0c1f012ea96bc9f460c57`
- **SVBench**: `2a89eaf00c5ac1cdd3abd717a9347052d39c4091`
- **OmniMMI**: `702488939108f057488dcb09fa0727e58b964174`
- **VideoMemBench**: `e4a76fd473c27fb914e6ea98365cc69d1d2cd7c9`

## 3. Frozen Video-Disjoint Dev Calibration
- `$\mu_B = 0.4219$`, `$\alpha_{\mathrm{ocr}} = 0.00$` (Dev ROC-AUC = `1.000`), `$\tau_{\mathrm{dig}} = 0.70$`
- Router `$T = 0.25$`, `$\gamma_{\mathrm{conf}} = 0.45$` (Dev ECE = `0.0000`), `$\tau_H = 0.9212$` (Dev Reranker Trigger Rate = `16.7%`)

## 4. Main Comparison at Matched Profile P1 (Mean ± 95% CI, 5 Seeds)
| Method | Single-Turn Avg | CG-IoU | Drill | Jump | Comp. | Silent | Multi-Turn Overall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Uniform FPS Sampling** | 32.2$\pm$12.0 | 0.0$\pm$0.0 | 37.8$\pm$8.0 | 0.0$\pm$0.0 | 24.8$\pm$19.0 | 24.0$\pm$23.3 | 36.6$\pm$6.3 |
| **Top-k CLIP/ASR** | 100.0$\pm$0.0 | 16.7$\pm$0.0 | 100.0$\pm$0.0 | 100.0$\pm$0.0 | 100.0$\pm$0.0 | 74.0$\pm$23.3 | 94.6$\pm$4.7 |
| **Static Visual Token Pruning** | 100.0$\pm$0.0 | 16.7$\pm$0.0 | 69.7$\pm$14.6 | 0.0$\pm$0.0 | 42.9$\pm$22.1 | 24.0$\pm$23.3 | 53.3$\pm$7.6 |
| **Pure Facility-Location** | 100.0$\pm$0.0 | 31.4$\pm$2.2 | 87.4$\pm$10.6 | 100.0$\pm$0.0 | 100.0$\pm$0.0 | 26.0$\pm$23.3 | 87.1$\pm$7.1 |
| **Offline Agentic Search** | 78.6$\pm$16.0 | 26.2$\pm$5.3 | 6.0$\pm$6.9 | 100.0$\pm$0.0 | 74.8$\pm$23.6 | 26.0$\pm$23.3 | 59.9$\pm$10.9 |
| **TurnFocus (Ours)** | 100.0$\pm$0.0 | 16.7$\pm$0.0 | 67.8$\pm$24.8 | 100.0$\pm$0.0 | 100.0$\pm$0.0 | 50.0$\pm$0.0 | 81.6$\pm$5.3 |

## 5. Off-Path HiGHS LP Optimality Audit (`\tilde{F}_u(S) / UB(S)`)
| Slice | Median | P10 | Worst |
| :--- | :---: | :---: | :---: |
| Profile P0 (tight budget) | 0.5423 | 0.5110 | 0.5024 |
| Profile P1 (balanced budget) | 0.4977 | 0.3054 | 0.2697 |
| Profile P2 (expanded budget) | 0.4967 | 0.3122 | 0.2712 |
| Intent: Global Summary | 0.9522 | 0.9144 | 0.7588 |
| Intent: Multi-Clip Comparison | 0.4977 | 0.4279 | 0.4274 |
| Intent: Local Drill-Down | 0.5137 | 0.2804 | 0.2697 |
| Overall Sampled Turns | 0.5212 | 0.3221 | 0.2697 |

## 6. Measured Wall-Clock Localization Latency (ms)
| Component | P50 (ms) | P95 (ms) |
| :--- | :---: | :---: |
| Intent & Coreference Router (ms) | 0.18 | 0.21 |
| O(M) Belief Filter & L0 Scoring (ms) | 2.27 | 2.32 |
| 1D NMS + Stratified Pooling (ms) | 1.55 | 1.74 |
| Lazy-Greedy Allocation, Alg. 1 (ms) | 4.64 | 5.60 |
| Optional 2B Reranker (Trig. 32.1%; cond. ms) | 1464.40 | 1476.33 |
| Total Online Front-End (amortized ms) | 8.98 | 1478.46 |
| Off-Path: LP Audit Solver (ms / sample) | 4.41 | 5.09 |
| One-Time Offline Index (s / hour video) | 4.17 | 4.50 |

## 7. Component-Wise Ablations at Profile P1 (Table 3)
| Variant | Sum. | Drill | Comp. | Silent | Avg. |
| :--- | :---: | :---: | :---: | :---: | :---: |
| TurnFocus (Full) | 100.0$\pm$0.0 | 67.8$\pm$24.8 | 100.0$\pm$0.0 | 50.0$\pm$0.0 | 81.6$\pm$5.3 |
| (i) w/o 1D Temporal NMS | 100.0$\pm$0.0 | 67.8$\pm$24.8 | 100.0$\pm$0.0 | 50.0$\pm$0.0 | 81.6$\pm$5.3 |
| (ii) w/o Stratified Summary Pool | 100.0$\pm$0.0 | 67.8$\pm$24.8 | 100.0$\pm$0.0 | 50.0$\pm$0.0 | 81.6$\pm$5.3 |
| (iii) Hard Intent Switching | 100.0$\pm$0.0 | 55.2$\pm$22.3 | 100.0$\pm$0.0 | 26.0$\pm$23.3 | 77.8$\pm$5.6 |
| (iv-a) Unbucketed Entity Coverage | 100.0$\pm$0.0 | 67.8$\pm$24.8 | 100.0$\pm$0.0 | 50.0$\pm$0.0 | 81.6$\pm$5.3 |
| (iv-b) Binary Entity Labels | 100.0$\pm$0.0 | 100.0$\pm$0.0 | 100.0$\pm$0.0 | 50.0$\pm$0.0 | 90.9$\pm$5.0 |
| (v) w/o DIG Fidelity Weighting | 100.0$\pm$0.0 | 87.4$\pm$10.6 | 100.0$\pm$0.0 | 26.0$\pm$23.3 | 87.1$\pm$7.1 |
| (vi) Uniform Reset (c_conf < gamma_conf) | 100.0$\pm$0.0 | 67.8$\pm$24.8 | 100.0$\pm$0.0 | 50.0$\pm$0.0 | 81.6$\pm$5.3 |
| (vii) Fixed Unclamped eta_mod | 100.0$\pm$0.0 | 94.0$\pm$6.9 | 100.0$\pm$0.0 | 65.3$\pm$25.6 | 92.8$\pm$5.5 |
