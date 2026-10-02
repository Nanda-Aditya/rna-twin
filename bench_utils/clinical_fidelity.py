"""Clinical phenotype fidelity metrics and plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bench_utils.plot_utils_ml import MODEL_COLORS


MISSING = "Missing"


def _columns(phtp0, phtp1, trgt, context_cols):
    columns = list(dict.fromkeys(column for column in context_cols if column != trgt))
    if not columns:
        raise ValueError("Clinical fidelity requires at least one non-target context column")
    missing = [
        column for column in (trgt, *columns)
        if column not in phtp0 or column not in phtp1
    ]
    if missing:
        raise KeyError(f"Clinical fidelity columns not found: {missing}")
    return columns


def _is_categorical(real, synthetic):
    combined = pd.concat([real, synthetic], ignore_index=True)
    if not pd.api.types.is_numeric_dtype(combined):
        return True
    values = pd.to_numeric(combined, errors="coerce").dropna().to_numpy()
    unique = np.unique(values)
    max_levels = min(10, max(2, int(np.sqrt(len(values)))))
    return (
        len(unique) <= max_levels
        and np.allclose(unique, np.round(unique))
    )


def compute_clinical_fidelity(phtp0, phtp1, trgt, context_cols):
    """Compare real and synthetic phenotypes within each target class."""
    columns = _columns(phtp0, phtp1, trgt, context_cols)
    target_levels = pd.unique(pd.concat([phtp0[trgt], phtp1[trgt]]).dropna())
    rows = []
    for target in target_levels:
        real_group = phtp0.loc[phtp0[trgt] == target]
        synthetic_group = phtp1.loc[phtp1[trgt] == target]
        for column in columns:
            real = real_group[column]
            synthetic = synthetic_group[column]
            if _is_categorical(real, synthetic):
                real = real.astype("string").fillna(MISSING)
                synthetic = synthetic.astype("string").fillna(MISSING)
                levels = pd.unique(pd.concat([real, synthetic], ignore_index=True))
                real_prevalence = real.value_counts(normalize=True)
                synthetic_prevalence = synthetic.value_counts(normalize=True)
                for level in levels:
                    real_value = float(real_prevalence.get(level, 0.0))
                    synthetic_value = float(synthetic_prevalence.get(level, 0.0))
                    rows.append({
                        "target_class": target, "variable": column,
                        "variable_type": "categorical", "level": level,
                        "real_value": real_value,
                        "synthetic_value": synthetic_value,
                        "metric": "absolute_prevalence_difference",
                        "metric_value": abs(synthetic_value - real_value),
                        "n_real": len(real), "n_synthetic": len(synthetic),
                    })
            else:
                real = pd.to_numeric(real, errors="coerce").dropna()
                synthetic = pd.to_numeric(synthetic, errors="coerce").dropna()
                pooled_sd = np.sqrt((real.var(ddof=1) + synthetic.var(ddof=1)) / 2)
                smd = (
                    (synthetic.mean() - real.mean()) / pooled_sd
                    if np.isfinite(pooled_sd) and pooled_sd > 0 else np.nan
                )
                rows.append({
                    "target_class": target, "variable": column,
                    "variable_type": "continuous", "level": "",
                    "real_value": real.mean(),
                    "synthetic_value": synthetic.mean(),
                    "metric": "standardized_mean_difference",
                    "metric_value": smd, "n_real": len(real),
                    "n_synthetic": len(synthetic),
                })
    return pd.DataFrame(rows)


def plot_clinical_fidelity(
    phtp0, phtp1, trgt, context_cols, metrics, output_path,
):
    """Plot real and synthetic phenotype distributions by target class."""
    columns = _columns(phtp0, phtp1, trgt, context_cols)
    target_levels = pd.unique(pd.concat([phtp0[trgt], phtp1[trgt]]).dropna())
    target_levels = target_levels[:3]
    real_color = MODEL_COLORS["real"]
    synthetic_color = MODEL_COLORS["dbtwin"]
    fig, axes = plt.subplots(
        len(columns), len(target_levels), squeeze=False,
        figsize=(5 * len(target_levels), 3.8 * len(columns)), dpi=300,
    )
    legend_added = False
    for row, column in enumerate(columns):
        categorical = _is_categorical(phtp0[column], phtp1[column])
        for col, target in enumerate(target_levels):
            ax = axes[row, col]
            real = phtp0.loc[phtp0[trgt] == target, column]
            synthetic = phtp1.loc[phtp1[trgt] == target, column]
            metric = metrics.loc[
                (metrics["target_class"].astype(str) == str(target))
                & (metrics["variable"] == column)
            ]
            if categorical:
                real = real.astype("string").fillna(MISSING)
                synthetic = synthetic.astype("string").fillna(MISSING)
                levels = pd.unique(pd.concat([real, synthetic], ignore_index=True))
                x = np.arange(len(levels))
                width = 0.38
                ax.bar(
                    x - width / 2,
                    [real.eq(level).mean() for level in levels], width,
                    label="Real", color=real_color, edgecolor="black",
                )
                ax.bar(
                    x + width / 2,
                    [synthetic.eq(level).mean() for level in levels], width,
                    label="Synthetic", color=synthetic_color, edgecolor="black",
                )
                ax.set(
                    ylabel=f"{column}\nprevalence" if row == 1 else "Prevalence",
                    xticks=x,
                    xticklabels=[str(level) for level in levels],
                )
                ax.tick_params(axis="x", rotation=30)
                score = metric["metric_value"].max()
                annotation = f"max |Δ prevalence| = {score:.2f}"
            else:
                values = [
                    pd.to_numeric(real, errors="coerce").dropna().to_numpy(),
                    pd.to_numeric(synthetic, errors="coerce").dropna().to_numpy(),
                ]
                if all(len(value) for value in values):
                    violins = ax.violinplot(values, showextrema=False)
                    for body, color in zip(violins["bodies"], (real_color, synthetic_color)):
                        body.set_facecolor(color)
                        body.set_edgecolor("black")
                        body.set_alpha(0.8)
                    ax.boxplot(values, widths=0.18, showfliers=False)
                ax.set(xticks=(1, 2), xticklabels=("Real", "Synthetic"),
                       ylabel=column.replace("[", "\n[", 1))
                score = metric["metric_value"].iloc[0] if len(metric) else np.nan
                annotation = f"SMD = {score:.2f}"
            ax.set_title(f"{target}\n{annotation}")
            ax.spines[["top", "right"]].set_visible(False)
            if categorical and not legend_added:
                ax.legend(frameon=False)
                legend_added = True
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
