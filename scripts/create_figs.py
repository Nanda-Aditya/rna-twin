#!/usr/bin/env python3
"""Render benchmark figures from saved numeric artifacts."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bench_utils.clinical_fidelity import plot_clinical_fidelity
from bench_utils.plot_utils_ml import shap_plots
from bench_utils.plot_utils_privacy import (
    plot_coexpression,
    plot_dcr_distributions,
    plot_saved_umap,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output_dir", type=Path)
parser.add_argument("--split")
args = parser.parse_args()
metrics = pd.read_csv(args.output_dir / "metrics.csv")
clinical_metrics = pd.read_csv(args.output_dir / "clinical_fidelity_metrics.csv")
clinical_data = pd.read_csv(args.output_dir / "clinical_fidelity_data.csv")

with np.load(args.output_dir / "benchmark_artifacts.npz", allow_pickle=False) as data:
    splits = data["splits"].tolist()
    split = args.split or splits[0]
    if split not in splits:
        raise ValueError(f"Unknown split {split}; available splits: {splits}")
    fold = splits.index(split) + 1
    prefix = f"f{fold}_"
    fig_dir = args.output_dir / "figs"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for index, (case, reference) in enumerate(data["contrasts"]):
        de_prefix = f"{prefix}c{index}_"
        real_lfc = data[f"{de_prefix}real_lfc"]
        synthetic_lfc = data[f"{de_prefix}synthetic_lfc"]
        selected = (
            (data[f"{de_prefix}real_padj"] < 0.05)
            & (data[f"{de_prefix}synthetic_padj"] < 0.10)
            & np.isfinite(real_lfc) & np.isfinite(synthetic_lfc)
        )
        fig, ax = plt.subplots(figsize=(5.2, 4), dpi=300)
        ax.scatter(real_lfc[selected], synthetic_lfc[selected],
                   s=8, alpha=0.5, color="#377eb8")
        ax.set(xlabel="Real log2 fold change",
               ylabel="Synthetic log2 fold change",
               title=f"{case} vs {reference}")
        ax.text(0.03, 0.97,
                f'Pearson r = {float(data[f"{de_prefix}lfc_pearson"]):.2f}',
                transform=ax.transAxes, va="top")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(fig_dir / f"de_{index}.png", bbox_inches="tight")
        plt.close(fig)

    plot_saved_umap(
        data[f"{prefix}umap_real"], data[f"{prefix}umap_synthetic"],
        data[f"{prefix}umap_real_target"], data[f"{prefix}umap_synthetic_target"],
        float(data[f"{prefix}umap_mmd"]), str(fig_dir))

    split_data = clinical_data.loc[clinical_data["split"].astype(str) == split]
    split_metrics = clinical_metrics.loc[
        clinical_metrics["split"].astype(str) == split
    ]
    target = str(data["target"])
    context_cols = data["context_cols"].tolist()
    plot_clinical_fidelity(
        split_data.loc[split_data["source"] == "real"],
        split_data.loc[split_data["source"] == "synthetic"],
        target, context_cols, split_metrics,
        fig_dir / "clinical_fidelity.png",
    )

    plot_coexpression(
        None, None, model_name="synthetic", fig_folder=str(fig_dir),
        real_matrix=data[f"{prefix}coexpression_real"],
        synthetic_matrix=data[f"{prefix}coexpression_synthetic"],
        rho=float(data[f"{prefix}coexpression_rho"]))

    p5 = metrics.loc[metrics["metric"] == "privacy_p5_ratio"].iloc[0]
    plot_dcr_distributions(
        [{"d_strn": data["dcr_synthetic_train"],
          "d_stst": data["dcr_synthetic_test"], "p5_ratio": p5["value"]}],
        "dbtwin", fig_folder=str(fig_dir), dcr_ratio=p5["value"],
        dcr_ratio_std=p5["std"], save_png=True)
    plt.close()

    real_shap = data[f"{prefix}real_shap"]
    synthetic_shap = data[f"{prefix}synthetic_shap"]
    class_labels = data["class_labels"].tolist()
    shap_plots(
        real_shap, synthetic_shap, str(fig_dir), data[f"{prefix}shap_genes"],
        model_name="Synthetic", top_k=min(9, len(data[f"{prefix}shap_genes"])),
        sp_k=50, r_k=50,
        map_genes=True,
        classes=range(len(class_labels)), orig_classes_=class_labels,
        sp_rho=data["shap_median_sp_rho"], sp_rho_std=data["shap_std_sp_rho"],
        pearson_r=data["shap_median_r"], pearson_r_std=data["shap_std_r"],
        save_png=True)
