"""DE gene-count bar chart of the benchmark report."""

import numpy as np
import matplotlib.pyplot as plt
from bench_utils.plot_utils_ml import STYLE, _style_ax


def de_bar_plots(real_de, synthetic_de, labels, colors, title=None):
    """
    Bar chart of DE gene counts; returns the figure.

    `real_de` and each mask in `synthetic_de` are boolean masks over the same genes. Real
    gets one bar (total); each synthetic mask gets its total and its recalled genes (also
    real DE, hatched), labelled with recall = recalled / real DE genes. `labels` and `colors`
    list the real bar first.
    """
    s = STYLE
    totals = [np.count_nonzero(mask) for mask in (real_de, *synthetic_de)]
    recalled = [np.count_nonzero(real_de & mask) for mask in synthetic_de]

    fig, ax = plt.subplots(
        figsize=(s["figsize"][0]*1.25 , s["figsize"][1]), dpi=300
    )
    x = np.arange(len(totals)) * 1.2
    w = 0.27
    bar_kw = dict(width=w, edgecolor="black", linewidth=0.5)

    ax.bar(x[0], totals[0], color=colors[0], **bar_kw)  # Real: single centered bar
    for i, (total, overlap, color) in enumerate(
            zip(totals[1:], recalled, colors[1:]), start=1):  # Synthetic: total + recalled
        ax.bar(
            x[i] - w / 2, total, color=color, alpha=0.8,
            label="Total" if i == 1 else "", **bar_kw,
        )
        ax.bar(
            x[i] + w / 2, overlap, color=color, alpha=0.5,
            label="Recalled" if i == 1 else "", hatch="//", **bar_kw,
        )
        ax.text(
            x[i] + w * 0.5 - 0.02,
            overlap + 0.063 * max(totals),
            f"{100 * overlap / totals[0]:.1f}%" if totals[0] else "N/A",
            fontsize=s["tick"]+3.6,fontweight="bold",
            color=color, ha="left", va="bottom")

    ax.set_ylabel("Number of DE genes", fontsize=s["label"])
    if title:
        ax.set_title(title, fontsize=s["title"])
    ax.set_xticks(x)
    ax.set_xticklabels(
        labels, fontsize=(s["tick"]+1) * 0.87, rotation=0,
        ha="center")
    ax.tick_params(axis='y', labelsize=s["tick"])
    ax.set_ylim(top=ax.get_ylim()[1] * 1.22)
    ax.legend(fontsize=s["legend"]+1, frameon=False,
              loc="upper left")
    _style_ax(ax)

    fig.tight_layout()
    return fig
