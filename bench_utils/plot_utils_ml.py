"""Shared benchmark plot style and colours, and the SHAP importance figures."""

from pathlib import Path

import matplotlib
import numpy as np
from bench_utils.de_ml_utils import map_ensembl_to_symbols
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Publication style: default figure size (inches) and font sizes of the benchmark figures
# ---------------------------------------------------------------------------
STYLE = {
    "figsize": (5.0, 4.0),
    "title": 16,
    "label": 15,
    "tick": 13.5,
    "legend": 14,
    "text": 14,
}

# One-time baseline rcParams for every bench_utils plotting module; no function updates them.
glb_fnt = 15
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": glb_fnt,
        "axes.labelsize": glb_fnt,
        "axes.titlesize": glb_fnt,
        "xtick.labelsize": glb_fnt,
        "ytick.labelsize": glb_fnt,
        "legend.fontsize": glb_fnt,
    }
)

# ---------------------------------------------------------------------------
# Model color scheme - consistent across all plots
# ---------------------------------------------------------------------------
MODEL_COLORS = {
    "real": "#333333",  # dark gray
    "dbtwin": "#35de92",  # light blue-greenish (main model)
}


def _style_ax(ax: plt.Axes) -> None:
    """Remove top/right spines and the grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def shap_plots(
    shap0,
    shap1,
    fig_folder,
    genes,
    class_labels,
    pearson_r,
    pearson_r_std,
    r_k,
    top_k,
    syn_color,
    model_name="Synthetic",
    split_label=None,
):
    """
    Grouped bar charts of mean |SHAP| for the `top_k` genes with the highest real importance.

    Parameters:
    -----------
    shap0, shap1 : array-like
        Real- and synthetic-trained SHAP values on the same test samples, shape
        (n_samples, n_genes) for a binary target or (n_samples, n_genes, n_classes)
    fig_folder : str or Path
        Output folder
    genes : array-like
        Gene IDs of the SHAP columns. Ensembl IDs are shown as HGNC symbols; Ensembl IDs
        without a symbol get blank tick labels
    class_labels : list
        Class labels in the order of the last SHAP axis (only the multiclass titles use them)
    pearson_r, pearson_r_std : float or array-like
        Mean and SD over splits of the real-vs-synthetic Pearson r of the top `r_k` genes;
        scalars for a binary target, one value per class otherwise
    top_k : int
        Number of genes plotted
    syn_color : color
        Synthetic bar color
    model_name : str
        Legend label of the synthetic bars and file-name stem
    split_label : str, optional
        Split shown in the titles

    One figure per class (one for a binary target) is saved to fig_folder as
    SHAP_{model_name}.pdf/.png, or SHAP_{model_name}_{class index}.pdf/.png.
    """
    shap0, shap1 = np.asarray(shap0), np.asarray(shap1)
    multiclass = shap0.ndim == 3
    gene_labels = np.asarray([
        " " if str(gene).startswith("ENSG") and str(label) == str(gene) else label
        for gene, label in zip(genes, map_ensembl_to_symbols(list(genes)))
    ])

    for cls_idx in range(len(class_labels) if multiclass else 1):
        mas0, mas1 = (
            np.nanmean(np.abs(values[:, :, cls_idx] if multiclass else values), axis=0)
            for values in (shap0, shap1))
        ix0 = np.argsort(np.where(np.isfinite(mas0), mas0, -np.inf))[::-1][:top_k]

        fig, ax = plt.subplots(figsize=STYLE["figsize"])
        xvec = np.arange(0, 4 * top_k, step=4)
        ax.bar(xvec, mas0[ix0], 0.89, label="Real", color=MODEL_COLORS["real"])
        ax.bar(xvec + 1.2, mas1[ix0], 0.89, label=model_name, color=syn_color)
        ax.set_xticks(xvec + 0.6)
        ax.set_xticklabels(gene_labels[ix0], rotation=35, ha="right", fontsize=STYLE["tick"])
        ax.set_ylabel("Mean |SHAP|", fontsize=STYLE["label"])
        ax.tick_params(axis="y", labelsize=STYLE["tick"])

        r, r_std = (np.atleast_1d(value)[cls_idx] for value in (pearson_r, pearson_r_std))
        title = rf"$r$ = ${np.round(r, 2)} \pm {np.round(r_std, 3)}$ (top {r_k})"
        if multiclass:
            title = f"{class_labels[cls_idx]}\n{title}"
        if split_label is not None:
            title += f"\nSplit {split_label}"
        ax.set_title(title)
        ax.legend(fontsize=STYLE["legend"], frameon=False)
        _style_ax(ax)

        fig.tight_layout()
        name = f"SHAP_{model_name}_{cls_idx}" if multiclass else f"SHAP_{model_name}"
        fig.savefig(Path(fig_folder) / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(Path(fig_folder) / f"{name}.png", bbox_inches="tight", dpi=300)
        plt.close(fig)
