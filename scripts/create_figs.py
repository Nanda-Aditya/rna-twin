#!/usr/bin/env python3
"""Render benchmark figures from saved numeric artifacts."""

import argparse
import json
from collections import Counter
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Resolve the helpers next to this script before editable installations.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_utils.config import (
    DE_LFC_MIN, DE_REAL_PADJ, DE_SYN_LFC_MIN, de_metric_rows, de_sets, load_gene_counts)
from bench_utils.clinical_fidelity import plot_clinical_fidelity, tvd_noise_band
from bench_utils.expression import log_cpm
from bench_utils.plot_utils_de import de_bar_plots
from bench_utils.plot_utils_ml import MODEL_COLORS, shap_plots
from bench_utils.plot_utils_privacy import (
    plot_attribute_inference,
    plot_coexpression,
    plot_dcr_distributions,
    plot_saved_umap,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output_dir", type=Path)
parser.add_argument("--split")
parser.add_argument("--bench-config", type=Path)
parser.add_argument("--input-dir", type=Path,
                    help="Real-data directory (default: <dataset>/real_data)")
args = parser.parse_args()
metrics = pd.read_csv(args.output_dir / "metrics.csv")
clinical_metrics = pd.read_csv(args.output_dir / "clinical_fidelity_metrics.csv")
clinical_data = pd.read_csv(args.output_dir / "clinical_fidelity_data.csv")

with np.load(args.output_dir / "benchmark_artifacts.npz", allow_pickle=False) as data:
    if not all(key in data for key in (
        "shap_mean_r", "shap_mean_sp_rho", "shap_raw_correlations_r",
        "shap_raw_correlations_sp", "shap_signal_failures")):
        raise ValueError("Saved SHAP metrics are outdated; rerun compute_metrics.py")
    if "full_de_genes" not in data:
        raise ValueError("Saved fidelity artifacts are fold-based; rerun compute_metrics.py")
    gene_counts = load_gene_counts(args.bench_config, data)
    splits = data["splits"].tolist()
    split = args.split or splits[0]
    if split not in splits:
        raise ValueError(f"Unknown split {split}; available splits: {splits}")
    fold = splits.index(split) + 1
    fig_dir = args.output_dir / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    def _save(fig, name):
        """Save a PDF for the report and a PNG for inspection, then close the figure."""
        for suffix in ("pdf", "png"):
            fig.savefig(fig_dir / f"{name}.{suffix}", bbox_inches="tight")
        plt.close(fig)

    contrast_pairs = [tuple(pair) for pair in data["contrasts"].tolist()]
    contrast_columns = (
        data["contrast_columns"].tolist() if "contrast_columns" in data
        else [str(data["target"])] * len(contrast_pairs)
    )
    input_dir = args.input_dir or args.output_dir.resolve().parents[1] / "real_data"
    expression = pd.read_csv(input_dir / "expression.csv.gz", index_col=0)
    gene_mask = data["full_de_expression_mask"]
    _, de_lfc_r = de_metric_rows(data, [
        (column, *pair) for column, pair in zip(contrast_columns, contrast_pairs)],
        gene_mask)
    pair_counts = Counter(contrast_pairs)
    dbtwin_color = np.asarray(matplotlib.colors.to_rgb(MODEL_COLORS["dbtwin"]))
    dbtwin_dark = tuple(dbtwin_color * 0.65)
    for index, ((case, reference), column) in enumerate(
        zip(contrast_pairs, contrast_columns)):
        real_lfc = data[f"full_c{index}_real_lfc"]
        synthetic_lfc = data[f"full_c{index}_synthetic_lfc"]
        real_de, synthetic_de = de_sets(data, index, gene_mask)
        selected = real_de & np.isfinite(synthetic_lfc)
        x, y = real_lfc[selected], synthetic_lfc[selected]
        r = de_lfc_r[index]
        slope, intercept = np.polyfit(x, y, 1) if len(x) > 2 else (np.nan, np.nan)
        real_label = f"(real padj < {DE_REAL_PADJ:g}, |log2FC| > {DE_LFC_MIN:g})"
        fig, ax = plt.subplots(figsize=(5.2, 4), dpi=300)
        ax.scatter(x, y, s=8, alpha=0.6, color=tuple(dbtwin_color),
                   edgecolors=dbtwin_dark, linewidths=0.3)
        lim = max(1.0, np.ceil(np.abs(np.r_[x, y]).max())) if len(x) else 1.0
        ax.plot([-lim, lim], [-lim, lim], ls="--", lw=1, color="#888888", label="Identity")
        if np.isfinite(slope):
            ax.plot([-lim, lim], np.polyval((slope, intercept), [-lim, lim]),
                    lw=1.6, color=dbtwin_dark, label="Least-squares fit")
        ax.legend(frameon=False, loc="upper left", fontsize=11)
        title = ((f"{column}: " if column != str(data["target"]) or
                  pair_counts[(case, reference)] > 1 else "")
                 + f"{case} vs {reference}")
        ax.set(
            xlim=(-lim, lim), ylim=(-lim, lim),
            xlabel=f"Real log2 fold change\n{real_label}",
            ylabel=f"Synthetic log2 fold change\n{real_label}",
            title=f"{title}\nPearson r = {r:.3f}, slope = {slope:.2f}",
        )
        if column not in data["anchor_cols"].tolist():
            ax.set_title(ax.get_title(), pad=24)
            ax.text(
                0.5, 1.02, "non-anchor feature", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=11, color="#e53935")
        fig.tight_layout()
        _save(fig, f"de_{index}")

        _save(de_bar_plots(
            real_de, list(synthetic_de.values()),
            labels=[f"Real\npadj < {DE_REAL_PADJ:.2f}\n|LFC| > {DE_LFC_MIN:g}", *[
                f"dbTwin ({label})\npadj < {threshold:g}\n"
                f"|LFC| > {DE_SYN_LFC_MIN[threshold]:g}"
                for label, threshold in zip(("strict", "relaxed"), synthetic_de)]],
            colors=["black", dbtwin_color * 0.65, dbtwin_color],
            title=title), f"gene_counts_{index}")

    target = str(data["target"])
    anchor_cols = [column for column in data["anchor_cols"].tolist()
                    if column != target]
    clinical_metrics = clinical_metrics.join(tvd_noise_band(
        *(clinical_data.loc[clinical_data["source"] == source]
          for source in ("real", "synthetic")), target, anchor_cols),
        on=["target_class", "variable"])
    for index, column in enumerate(anchor_cols):
        _save(plot_clinical_fidelity(
            clinical_data.loc[clinical_data["source"] == "real"],
            clinical_data.loc[clinical_data["source"] == "synthetic"],
            target, [column], clinical_metrics,
        ), f"clinical_fidelity{index}")

    fig, (umap_ax, coexp_ax) = plt.subplots(
        1, 2, figsize=(10.5, 4), dpi=300,
        gridspec_kw={"width_ratios": [1.2, 0.75]})
    plot_saved_umap(
        data["full_umap_real"], data["full_umap_synthetic"],
        data["full_umap_real_target"], data["full_umap_synthetic_target"],
        float(data["full_umap_mmd"]), umap_ax, legend_scale=1.8)
    rho = float(data["full_coexpression_rho"])
    plot_coexpression(
        data["full_coexpression_real"], data["full_coexpression_synthetic"], coexp_ax)
    coexp_ax.text(
        0.5, 1.02, "Real: lower triangle\nSynthetic: upper triangle",
        transform=coexp_ax.transAxes, ha="center", va="bottom", fontsize=11)
    for ax, label in ((umap_ax, "A"), (coexp_ax, "B")):
        ax.set_title(label, loc="left", fontweight="bold")
    coexp_ax.set_title(rf"Spearman $\rho$ = {rho:.2g}", pad=38, x=0.6)
    # Rasterize dense UMAP points and heat-map cells so the PDF stays small.
    for artist in (*umap_ax.collections, *coexp_ax.collections, *coexp_ax.images):
        artist.set_rasterized(True)
    fig.tight_layout()
    _save(fig, "expression_fidelity")

    # Appendix: per-gene log2(CPM + 1) statistics and library size, full real vs full release.
    dataset_dir = input_dir.resolve()
    dataset_name = (dataset_dir.parent.name if dataset_dir.name == "real_data"
                    else dataset_dir.name)
    full_syn = pd.read_csv(args.output_dir.parent / f"syn_{dataset_name}.csv", index_col=0)
    stats = {}
    for source, counts in (("real", expression),
                           ("synthetic", full_syn[expression.columns])):
        expression_cpm = log_cpm(counts)
        stats[source] = {"mean": expression_cpm.mean(), "variance": expression_cpm.var(),
                         "zero": (counts == 0).mean(), "library": counts.sum(axis=1) / 1e6}
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), dpi=300)
    for ax, (key, label) in zip(axes.flat, (
            ("mean", "Mean log2(CPM + 1)"), ("variance", "Variance of log2(CPM + 1)"),
            ("zero", "Fraction of zero counts"))):
        x, y = stats["real"][key], stats["synthetic"][key]
        lim = (0, 1.03 * max(x.max(), y.max()))
        ax.scatter(x, y, s=3, alpha=0.3, color=dbtwin_dark, edgecolors="none",
                   rasterized=True)
        ax.plot(lim, lim, ls="--", lw=1, color="#888888")
        subtitle = f"Pearson r = {x.corr(y):.3f}"
        if key == "variance":
            subtitle += f", median ratio = {(y[x > 0] / x[x > 0]).median():.2f}"
        ax.set(xlim=lim, ylim=lim, aspect="equal", xlabel="Real (per gene)",
               ylabel="Synthetic (per gene)", title=f"{label}\n{subtitle}")
    ax = axes[1, 1]
    libraries = [stats[source]["library"].dropna().to_numpy() for source in stats]
    violins = ax.violinplot(libraries, widths=0.5, showextrema=False)
    for body, color in zip(violins["bodies"], (MODEL_COLORS["real"], MODEL_COLORS["dbtwin"])):
        body.set(facecolor=color, edgecolor="black", alpha=0.8)
    ax.boxplot(libraries, widths=0.18, showfliers=False, medianprops={"color": "black"})
    ax.set(xticks=(1, 2), xticklabels=("Real", "Synthetic"),
           ylabel="Total counts per sample (millions)",
           title="Library size\nmedian {:.1f}M real, {:.1f}M synthetic".format(
               *map(np.median, libraries)))
    for ax, label in zip(axes.flat, "ABCD"):
        ax.text(-0.2, 1.2, label, transform=ax.transAxes, fontweight="bold", va="top")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, "gene_stats")

    fig, (dcr_ax, attribute_ax) = plt.subplots(
        1, 2, figsize=(10.5, 4), dpi=300,
        gridspec_kw={"width_ratios": [43, 57]})
    p5 = metrics.loc[metrics["metric"] == "privacy_p5_ratio"].iloc[0]
    plot_dcr_distributions(
        data["dcr_syn_trn"], data["dcr_trn_trn"], p5["value"], p5["std"], dcr_ax)
    plot_attribute_inference(
        pd.read_csv(args.output_dir / "attribute_inference.csv"),
        gene_counts["n_genes_aux"], attribute_ax)
    attribute_ax.set_ylim(0, 0.8)
    for ax, label in ((dcr_ax, "A"), (attribute_ax, "B")):
        ax.set_title(label, loc="left", fontweight="bold")
    fig.tight_layout()
    _save(fig, "privacy")

    # Per-split TRTR and TSTR AUC on the same held-out real fold; dashed lines are means.
    positions = np.arange(len(splits))
    aucs = {kind: np.array([float(data[f"f{k}_{kind}_auc"])
                            for k in range(1, len(splits) + 1)])
            for kind in ("real", "synthetic")}
    fig, ax = plt.subplots(figsize=(5.6, 3.2), dpi=300)
    ax.plot([positions - 0.1, positions + 0.1], [aucs["real"], aucs["synthetic"]],
            color="#BBBBBB", lw=1, zorder=1)
    for kind, label, color, dx in (
            ("real", "TRTR (real-trained)", MODEL_COLORS["real"], -0.1),
            ("synthetic", "TSTR (synthetic-trained)", MODEL_COLORS["dbtwin"], 0.1)):
        ax.axhline(aucs[kind].mean(), ls="--", lw=1, color=color, zorder=0)
        ax.scatter(positions + dx, aucs[kind], s=55, color=color, edgecolors="black",
                   linewidths=0.5, label=label, zorder=3)
    low, high = min(map(np.min, aucs.values())), max(map(np.max, aucs.values()))
    ax.set(xticks=positions, xticklabels=splits, xlabel="Outer split",
           ylabel="ROC-AUC", ylim=(low - 0.02, high + 0.02))
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1), ncol=2,
              fontsize=12, handletextpad=0.2, columnspacing=1)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, "auc_folds")

    valid_shap = (
        np.isfinite(data["shap_raw_correlations_r"]).reshape(len(splits), -1).all(axis=1)
        & np.isfinite(data["shap_raw_correlations_sp"]).reshape(len(splits), -1).all(axis=1))

    def _nonzero_shap(fold):
        """Whether the plotted top real genes have any non-zero real and synthetic SHAP per class."""
        real, synthetic = (
            np.nanmean(np.abs(data[f"f{fold + 1}_{kind}_shap"]), axis=0)
            for kind in ("real", "synthetic"))
        real, synthetic = real.reshape(len(real), -1), synthetic.reshape(len(real), -1)
        top = np.argsort(np.where(np.isfinite(real), real, -np.inf), axis=0)[::-1][:9]
        return bool((np.take_along_axis(real, top, 0) > 0).any(axis=0).all()
                    and (np.take_along_axis(synthetic, top, 0) > 0).any(axis=0).all())

    valid_shap &= [_nonzero_shap(i) for i in range(len(splits))]
    shap_fold = fold - 1 if valid_shap[fold - 1] else next(
        (i for i, valid in enumerate(valid_shap) if valid), None)
    shap_split = splits[shap_fold] if shap_fold is not None else None
    (args.output_dir / "figure_metadata.json").write_text(
        json.dumps({"figure_split": split, "shap_split": shap_split}), encoding="utf-8")
    if shap_fold is None:
        print("No split has valid SHAP correlations and non-zero plotted SHAP values "
              "for every class; omitting SHAP panels.")
    else:
        shap_prefix = f"f{shap_fold + 1}_"
        class_labels = data["class_labels"].tolist()
        shap_plots(
            data[f"{shap_prefix}real_shap"], data[f"{shap_prefix}synthetic_shap"],
            fig_dir, data[f"{shap_prefix}shap_genes"], class_labels,
            data["shap_mean_r"], data["shap_std_r"], r_k=50,
            top_k=min(9, len(data[f"{shap_prefix}shap_genes"])),
            syn_color=MODEL_COLORS["dbtwin"], split_label=shap_split)
