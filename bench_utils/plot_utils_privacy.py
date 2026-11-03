"""Privacy and global expression panels: DCR, attribute inference, co-expression and UMAP.

Each function draws on an existing axis of a multi-panel figure in create_figs.py.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from bench_utils.de_ml_utils import map_ensembl_to_symbols
from bench_utils.plot_utils_ml import STYLE, MODEL_COLORS


TEXT_COLOR = "#333333"


def _apply_theme(ax, fontsize_tick=12):
    """Minimal, print-friendly axis styling."""
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#AAAAAA")
    ax.spines["bottom"].set_color("#AAAAAA")
    ax.tick_params(axis="both", labelsize=fontsize_tick, colors=TEXT_COLOR)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, integer=False))
    ax.grid(False)


# ---------------------------------------------------------------------------
# Color blending helpers for the two-group privacy panels
# ---------------------------------------------------------------------------
def _blend_colors(base_hex: str, shade_hex: str, intensity: float = 0.25) -> str:
    """Hex color `intensity` of the way from `base_hex` to `shade_hex` (0 = base, 1 = shade)."""
    base, shade = ([int(color[i:i+2], 16) for i in (1, 3, 5)] for color in (base_hex, shade_hex))
    return "#" + "".join(
        f"{int(b * (1 - intensity) + s * intensity):02x}" for b, s in zip(base, shade))


def _get_dcr_colors(model_name: str, shade_intensity: float = 0.25) -> tuple:
    """
    The model color blended toward blue, then toward red: the color pair of both privacy panels.

    DCR uses blue for synthetic-to-training and red for training-to-training violins;
    attribute inference uses blue for the clinical-only and red for the expression-aided attack.
    """
    base_color = MODEL_COLORS.get(model_name, "#666666")
    return tuple(_blend_colors(base_color, shade, intensity=shade_intensity)
                 for shade in ("#3366CC", "#CC3333"))


# ---------------------------------------------------------------------------
# DCR distributions  (violin of per-sample DCR, not per-run p5)
# ---------------------------------------------------------------------------
def plot_dcr_distributions(d_st, d_tt, p5_ratio, p5_ratio_std, ax, model_name="dbtwin"):
    """
    Side-by-side violins of synthetic-to-training (`d_st`) and training-to-training (`d_tt`)
    distances to the closest record, pooled over all splits. The annotation gives the p5
    ratio as mean ± SD over splits.
    """
    s = STYLE
    trn_syn_color, trn_trn_color = _get_dcr_colors(model_name)

    # --- side-by-side violins ---
    parts = ax.violinplot(
        [np.ravel(d_st), np.ravel(d_tt)],
        positions=[0, 1],
        showmedians=True,
        showextrema=False,
        widths=0.6,
    )
    for body, color in zip(parts["bodies"], (trn_syn_color, trn_trn_color)):
        body.set_facecolor(color)
        body.set_alpha(0.35)
    parts["cmedians"].set_color("black")

    # --- annotation ---
    ax.text(
        0.65,
        0.95,
        f"p5 ratio = {p5_ratio:.2f} ± {p5_ratio_std:.2f}",
        transform=ax.transAxes,
        fontsize=(s["text"]+0.2) * 0.75,
        fontweight="bold",
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc"),
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["train-synth", "train-train"], fontsize=(s["tick"]+2.5) * 0.75)
    ax.set_ylabel("DCR on scaled log-CPM", fontsize=s["label"] * 0.75)
    _apply_theme(ax, fontsize_tick=s["tick"] * 0.75)


def plot_attribute_inference(attribute, n_genes_aux, ax):
    """Box plots of splitwise excess attribute-inference success for the top 7 secrets.

    `attribute` holds attribute_inference.csv rows and is not modified. Excess is training
    minus control attack success; the 7 secrets with the highest mean excess (over both
    attacks and all splits) are shown. Boxes span splits (whiskers to the full range) and dots
    are splits, for the clinical-QI attack and the QI + `n_genes_aux` genes attack. Gene
    secrets are labelled with offline HGNC symbols (`map_ensembl_to_symbols`).
    """
    attribute = attribute.assign(
        excess=attribute["training_attack_success_rate"]
        - attribute["control_attack_success_rate"]
    )
    ranked_secrets = (
        attribute.groupby("secret")["excess"].mean().nlargest(7).index.tolist()
    )
    dcr_train_color, dcr_test_color = _get_dcr_colors("dbtwin")
    for attack, offset, color, label in (
        ("clinical_qi", -0.16, dcr_train_color, "Clinical QI"),
        ("standard", 0.16, dcr_test_color, f"Clinical QI + {n_genes_aux} genes"),
    ):
        values = [
            attribute.loc[
                (attribute["secret"] == secret) & (attribute["attack"] == attack),
                "excess",
            ].dropna().to_numpy()
            for secret in ranked_secrets
        ]
        available = [(index + 1 + offset, fold_values) for index, fold_values
                     in enumerate(values) if len(fold_values)]
        if available:
            boxes = ax.boxplot(
                [fold_values for _, fold_values in available],
                positions=[position for position, _ in available],
                widths=0.26, patch_artist=True, manage_ticks=False, whis=(0, 100),
                medianprops={"color": "#333333"},
            )
            for box in boxes["boxes"]:
                box.set(facecolor=color, edgecolor="#333333", linewidth=0.8)
            for position, fold_values in available:
                ax.scatter(np.full(len(fold_values), position), fold_values,
                           color="#333333", s=7, zorder=3)
        ax.plot([], [], color=color, linewidth=7, label=label)
    ax.set_xticks(range(1, len(ranked_secrets) + 1), [
        (map_ensembl_to_symbols([secret])[0]
         if str(secret).startswith("ENSG") else
         secret.removeprefix("Sample Characteristic[").removesuffix("]"))
        for secret in ranked_secrets
    ], rotation=45, ha="right", fontsize=10)
    ax.set(xlim=(0.5, len(ranked_secrets) + 0.5), ylim=(0, 0.5),
           ylabel="Training - control success")
    ax.yaxis.label.set_size(12)
    ax.tick_params(axis="y", labelsize=10)
    ax.legend(frameon=False, fontsize=11, loc="upper right")
    ax.grid(axis="y", alpha=0.2)


def plot_coexpression(real_matrix, synthetic_matrix, ax):
    """
    Heat map of gene-gene correlations: real below the diagonal, synthetic above it.

    Both matrices must share one gene order, e.g. `ordered_genes` from
    compute_coexpression_artifacts (hierarchical clustering of the real matrix), so preserved
    modules show as mirror-image blocks. The symmetric color range is the 85th percentile of
    absolute real off-diagonal correlations; the color bar sits right of `ax`. The caller
    sets the title.
    """
    c_real, c_syn = np.asarray(real_matrix), np.asarray(synthetic_matrix)
    n = len(c_real)

    # Lower triangle and diagonal = real, upper triangle = synthetic
    combined = np.where(np.tril(np.ones((n, n), dtype=bool), k=-1), c_real, c_syn)
    np.fill_diagonal(combined, np.diag(c_real))

    vabs = np.percentile(np.abs(c_real[np.triu_indices(n, k=1)]), 85)
    im = ax.imshow(combined, cmap="RdBu_r", vmin=-vabs, vmax=vabs, aspect="equal")
    cbar = ax.figure.colorbar(im, cax=ax.inset_axes([1.02, 0.1, 0.035, 0.8]))
    cbar.ax.tick_params(labelsize=STYLE["tick"])

    ax.plot([0, n - 1], [0, n - 1], "k-", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Genes", fontsize=STYLE["label"])
    ax.set_ylabel("Genes", fontsize=STYLE["label"])
    ax.set_xticks([])
    ax.set_yticks([])


# Okabe-Ito, ordered so the RISK classes take the colours of the paper's Fig. 2D.
CLASS_COLORS = ("#D55E00", "#56B4E9", "#009E73", "#E69F00", "#CC79A7", "#0072B2", "#F0E442")


def plot_saved_umap(real, synthetic, real_target, synthetic_target, mmd, ax, legend_scale=1):
    """Scatter saved joint UMAP coordinates coloured by target class, real filled and synthetic
    open; missing classes are grey and the title gives the MMD."""
    classes = sorted(set(real_target) | set(synthetic_target), key=lambda c: (c == "nan", c))
    palette = CLASS_COLORS if len(classes) <= len(CLASS_COLORS) else plt.cm.tab20.colors
    handles = []
    for cls, color in zip(classes, palette):
        color = "#AAAAAA" if cls == "nan" else color
        ax.scatter(*real[real_target == cls].T, s=22, facecolor=color, edgecolor="none", alpha=0.8)
        ax.scatter(*synthetic[synthetic_target == cls].T, s=22, facecolor="none", edgecolor=color,
                   linewidths=0.8, alpha=0.8)
        handles.append(Line2D([], [], ls="", marker="o", color=color,
                              label="missing" if cls == "nan" else cls))
    handles += [Line2D([], [], ls="", marker="o", color="#777777", label="Real"),
                Line2D([], [], ls="", marker="o", mfc="none", color="#777777", label="Synthetic")]
    ax.set(xlabel="UMAP 1", ylabel="UMAP 2", title=f"MMD = {mmd:.3g}")
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False,
              fontsize=7 * legend_scale, markerscale=0.7 * legend_scale,
              handletextpad=0.3,
              borderaxespad=0.2)
