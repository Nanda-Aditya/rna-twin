"""Clinical phenotype fidelity metrics and plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bench_utils.plot_utils_ml import MODEL_COLORS


MISSING = "Missing"


def _columns(phtp0, phtp1, trgt, anchor_cols):
    """Unique anchor columns other than `trgt`, checked to exist in both frames."""
    columns = list(dict.fromkeys(column for column in anchor_cols if column != trgt))
    if not columns:
        raise ValueError("Clinical fidelity requires at least one non-target anchor column")
    missing = [
        column for column in (trgt, *columns)
        if column not in phtp0 or column not in phtp1
    ]
    if missing:
        raise KeyError(f"Clinical fidelity columns not found: {missing}")
    return columns


def _is_categorical(real):
    """Whether a variable is compared as categorical, decided once on its full real cohort.

    Non-numeric variables are categorical. Numeric ones are categorical only when every value
    is an integer and there are at most min(10, max(2, floor(sqrt(n)))) distinct values, n being
    the number of non-missing values; otherwise they are continuous.
    """
    if not pd.api.types.is_numeric_dtype(real):
        return True
    values = pd.to_numeric(real, errors="coerce").dropna().to_numpy()
    unique = np.unique(values)
    max_levels = min(10, max(2, int(np.sqrt(len(values)))))
    return (
        len(unique) <= max_levels
        and np.allclose(unique, np.round(unique))
    )


def tvd(real, synth, numeric=False, n_bins=10):
    """Total variation distance; numeric values use `n_bins` equal-width bins defined on `real`."""
    real, synth = pd.Series(real).dropna(), pd.Series(synth).dropna()
    if numeric:
        edges = np.histogram_bin_edges(real, bins=n_bins)
        # widen outer edges so out-of-range synthetic values are counted
        edges[0], edges[-1] = -np.inf, np.inf
        real, synth = pd.cut(real, edges), pd.cut(synth, edges)
    p, q = real.value_counts(normalize=True).align(
        synth.value_counts(normalize=True), fill_value=0)
    return 0.5 * np.abs(p - q).sum()


def compute_clinical_fidelity(phtp0, phtp1, trgt, anchor_cols):
    """Compare real (`phtp0`) and synthetic (`phtp1`) phenotypes within each target class.

    For each anchor column other than `trgt`, computes the total variation distance (TVD)
    between real and synthetic distributions. Categorical variables (`_is_categorical` on the
    full real cohort, so every class uses the same type) keep missing values as their own
    level; continuous variables use 10 fixed-width bins defined on the real class. Returns one
    row per (target class, variable).
    """
    columns = _columns(phtp0, phtp1, trgt, anchor_cols)
    categorical_columns = {column: _is_categorical(phtp0[column]) for column in columns}
    target_levels = pd.unique(pd.concat([phtp0[trgt], phtp1[trgt]]).dropna())
    rows = []
    for target in target_levels:
        real_group = phtp0.loc[phtp0[trgt] == target]
        synthetic_group = phtp1.loc[phtp1[trgt] == target]
        for column in columns:
            categorical = categorical_columns[column]
            real, synthetic = (
                frame[column].astype("string").fillna(MISSING) if categorical
                else pd.to_numeric(frame[column], errors="coerce").dropna()
                for frame in (real_group, synthetic_group))
            rows.append({
                "target_class": target, "variable": column,
                "variable_type": "categorical" if categorical else "continuous",
                "metric": "total_variation_distance",
                "metric_value": tvd(real, synthetic, numeric=not categorical),
                "n_real": len(real), "n_synthetic": len(synthetic),
            })
    return pd.DataFrame(rows)


def tvd_noise_band(phtp0, phtp1, trgt, anchor_cols, n_boot=500, seed=0):
    """TVD expected from sampling alone, per (target class, variable).

    Each real class is compared with `n_boot` bootstrap resamples of itself, each the size of
    the matching synthetic class; returns their 2.5th and 97.5th percentiles (null_lo, null_hi).
    """
    sizes = phtp1[trgt].value_counts()
    rng = np.random.default_rng(seed)
    null = pd.concat([compute_clinical_fidelity(phtp0, pd.concat(
        [group.sample(sizes.get(level, 0), replace=True, random_state=rng)
         for level, group in phtp0.groupby(trgt)]), trgt, anchor_cols)
        for _ in range(n_boot)])
    band = null.groupby(["target_class", "variable"])["metric_value"].quantile([0.025, 0.975])
    return band.unstack().set_axis(["null_lo", "null_hi"], axis=1)


def plot_clinical_fidelity(phtp0, phtp1, trgt, anchor_cols, metrics):
    """Plot real and synthetic phenotype distributions by target class; returns the figure.

    One row per anchor column and one panel per target class, for at most the first 3 classes.
    Variable types match `compute_clinical_fidelity`. Titles give the TVD from `metrics`;
    when `metrics` has a `null_hi` column (`tvd_noise_band`), they also give that bound and
    whether the observed TVD is within sampling noise.
    """
    columns = _columns(phtp0, phtp1, trgt, anchor_cols)
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
        categorical = _is_categorical(phtp0[column])
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
                width = 0.38 * 0.60
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
                    ylabel=f"{column}\nprevalence",
                    xticks=x,
                    xticklabels=[str(level) for level in levels],
                )
                ax.tick_params(axis="x", rotation=30)
            else:
                values = [
                    pd.to_numeric(real, errors="coerce").dropna().to_numpy(),
                    pd.to_numeric(synthetic, errors="coerce").dropna().to_numpy(),
                ]
                if all(len(value) for value in values):
                    violins = ax.violinplot(values, widths=0.5 * 0.80, showextrema=False)
                    for body, color in zip(violins["bodies"], (real_color, synthetic_color)):
                        body.set_facecolor(color)
                        body.set_edgecolor("black")
                        body.set_alpha(0.8)
                        body.set_label("Real" if color == real_color else "Synthetic")
                    ax.boxplot(values, widths=0.18, showfliers=False)
                ax.set(xticks=(1, 2), xticklabels=("Real", "Synthetic"),
                       ylabel=column.replace("[", "\n[", 1))
            score = metric["metric_value"].iloc[0] if len(metric) else np.nan
            bottom, top = ax.get_ylim()
            ax.set_ylim(bottom, top + 0.55 * (top - bottom))
            title = f"target = {target}\nTVD = {score:.2f}"
            if "null_hi" in metric and len(metric):
                noise = metric["null_hi"].iloc[0]
                title += (f" (noise ≤ {noise:.2f})\n"
                          f"within noise: {'yes' if score <= noise else 'no'}")
            ax.set_title(title)
            ax.spines[["top", "right"]].set_visible(False)
            if not legend_added:
                ax.legend(frameon=False, loc="upper center", ncol=2)
                legend_added = True
    fig.tight_layout()
    return fig
