"""Automated LaTeX table and pgfplots coordinate exporter from measured JSON results."""

from __future__ import annotations

from typing import Any, Dict, List


def fmt_mean_ci(mean_pct: float, ci_pct: float) -> str:
    return f"{mean_pct:.1f}$\\pm${ci_pct:.1f}"


def export_pgfplots_coordinates(pareto_dict: Dict[str, List[Dict[str, float]]]) -> str:
    """Export drop-in \\addplot coordinates with 95% CI error bars for Figure 3 (fig:pareto_dig)."""
    lines: List[str] = ["% Auto-generated from measured empirical runs (mean +- 95% CI)"]
    style_map = {
        "TurnFocus (Ours)": "color=blue!70!black, mark=*, thick",
        "Pure Facility-Location": "color=violet!70!black, mark=square*, thick, densely dashed",
        "Top-k CLIP/ASR": "color=teal!70!black, mark=triangle*, thick, dotted",
        "Static Visual Token Pruning": "color=orange!80!black, mark=diamond*, thick, dashdotted",
        "Uniform FPS Sampling": "color=gray!70!black, mark=x, thick",
    }
    for method_name, pts in pareto_dict.items():
        style = style_map.get(method_name, "thick, mark=*")
        coords = " ".join(
            f"({idx+1}, {pt['mean']:.2f}) +- (0, {pt['ci95']:.2f})" for idx, pt in enumerate(pts)
        )
        lines.append(
            f"% {method_name}\n"
            f"\\addplot+[{style}, error bars/.cd, y dir=both, y explicit] coordinates {{{coords}}};"
        )
    return "\n".join(lines) + "\n"


def export_markdown_and_latex_summary(results: Dict[str, Any]) -> str:
    """Format human-readable and LaTeX-ready empirical summary report with explicit scope notice."""
    out: List[str] = [
        "# TurnFocus Controlled Synthetic GPU Sanity-Check Report",
        "",
        "> **Scope Notice (Synthetic Sanity Check — Not Real Public Benchmarks)**: "
        "This report logs end-to-end execution of real unmodified HuggingFace model weights "
        "(`Qwen/Qwen3-VL-8B-Instruct`, `Qwen/Qwen3-VL-2B-Instruct`, `openai/clip-vit-base-patch32`) "
        "on a small controlled synthetic audiovisual testbed (4 Dev / 15 Test video-disjoint template episodes). "
        "It is intended for functional, LP-audit, and wall-clock latency verification and **must not** be "
        "interpreted as full evaluation on the 8 real public long-video benchmarks.",
        "",
        "## 1. Hardware & Pinned Checkpoint Provenance",
    ]
    prov = results.get("provenance", {})
    out.append(f"- **Timestamp (UTC)**: `{prov.get('timestamp_utc', 'N/A')}`")
    out.append(f"- **GPU Hardware**: `{prov.get('gpu_name', 'CPU')}` ({prov.get('gpu_vram_gb', 0)} GB VRAM)")
    out.append(f"- **Stack**: Python `{prov.get('python_version')}`, PyTorch `{prov.get('torch_version')}`, CUDA `{prov.get('cuda_version')}`, Transformers `{prov.get('transformers_version')}`")
    if "pinned_model_shas" in prov:
        for k, v in prov["pinned_model_shas"].items():
            out.append(f"- **Pinned Checkpoint `{k}`**: `{v}`")

    if "hf_dataset_shas" in results:
        out.append("\n## 2. Live HuggingFace Benchmark Registry SHAs")
        for b_name, b_sha in results["hf_dataset_shas"].items():
            out.append(f"- **{b_name}**: `{b_sha}`")

    if "calibration" in results:
        cal = results["calibration"]
        out.append("\n## 3. Frozen Video-Disjoint Dev Calibration")
        out.append(
            f"- `$\\mu_B = {cal['mu_b']:.4f}$`, `$\\alpha_{{\\mathrm{{ocr}}}} = {cal['alpha_ocr']:.2f}$` "
            f"(Dev ROC-AUC = `{cal['dig_roc_auc']:.3f}`), `$\\tau_{{\\mathrm{{dig}}}} = {cal['tau_dig']:.2f}$`"
        )
        out.append(
            f"- Router `$T = {cal['router_temperature']:.2f}$`, `$\\gamma_{{\\mathrm{{conf}}}} = {cal['gamma_conf']:.2f}$` "
            f"(Dev ECE = `{cal['router_dev_ece']:.4f}`), `$\\tau_H = {cal['tau_h']:.4f}$` "
            f"(Dev Reranker Trigger Rate = `{cal['dev_reranker_trigger_rate']*100:.1f}%`)"
        )

    if "main_comparison" in results:
        out.append("\n## 4. Main Comparison at Matched Profile P1 (Mean ± 95% CI, 5 Seeds)")
        out.append("| Method | Single-Turn Avg | CG-IoU | Drill | Jump | Comp. | Silent | Multi-Turn Overall |")
        out.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for m_name, row in results["main_comparison"].items():
            out.append(
                f"| **{m_name}** | {fmt_mean_ci(*row['single_avg'])} | {fmt_mean_ci(*row['cg_iou'])} | "
                f"{fmt_mean_ci(*row['drill'])} | {fmt_mean_ci(*row['jump'])} | {fmt_mean_ci(*row['comp'])} | "
                f"{fmt_mean_ci(*row['silent'])} | {fmt_mean_ci(*row['overall'])} |"
            )

    if "lp_audit" in results:
        out.append("\n## 5. Off-Path HiGHS LP Optimality Audit (`\\tilde{F}_u(S) / UB(S)`)")
        out.append("| Slice | Median | P10 | Worst |")
        out.append("| :--- | :---: | :---: | :---: |")
        for sl_name, stats_d in results["lp_audit"].items():
            out.append(
                f"| {sl_name} | {stats_d['median']:.4f} | {stats_d['p10']:.4f} | {stats_d['worst']:.4f} |"
            )

    if "latency_ms" in results:
        out.append("\n## 6. Measured Wall-Clock Localization Latency (ms)")
        out.append("| Component | P50 (ms) | P95 (ms) |")
        out.append("| :--- | :---: | :---: |")
        for comp_name, (p50, p95) in results["latency_ms"].items():
            out.append(f"| {comp_name} | {p50:.2f} | {p95:.2f} |")

    if "ablations" in results:
        out.append("\n## 7. Component-Wise Ablations at Profile P1 (Table 3)")
        out.append("| Variant | Sum. | Drill | Comp. | Silent | Avg. |")
        out.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
        for v_name, row in results["ablations"].items():
            out.append(
                f"| {v_name} | {fmt_mean_ci(*row['sum'])} | {fmt_mean_ci(*row['drill'])} | "
                f"{fmt_mean_ci(*row['comp'])} | {fmt_mean_ci(*row['silent'])} | {fmt_mean_ci(*row['avg'])} |"
            )

    return "\n".join(out) + "\n"
