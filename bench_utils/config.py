"""Benchmark settings and the DE gene definition shared by metrics, figures and reports."""

# Debug mode: Set to 1 for fast iteration with reduced max_iter values
demo_run = 0  # Default: 0 for production, 1 for debug/fast mode

# DE gene definition shared by figures and reports: real DE genes use DE_REAL_PADJ;
# synthetic DE genes use paired strict and relaxed padj and |log2FC| cutoffs.
DE_REAL_PADJ = 0.05
DE_SYN_PADJ = (0.05, 0.075)
DE_LFC_MIN = 0.5
DE_SYN_LFC_MIN = {0.05: 0.5, 0.075: 0.4}


def de_sets(artifacts, index, gene_mask=None):
    """Return the real DE mask and {synthetic threshold: mask} for full-cohort contrast `index`.

    Genes with missing adjusted p-values or fold changes are not DE.
    """
    import numpy as np

    def mask(kind, padj, lfc_min):
        p = np.nan_to_num(artifacts[f"full_c{index}_{kind}_padj"], nan=1.0)
        lfc = np.nan_to_num(artifacts[f"full_c{index}_{kind}_lfc"], nan=0.0)
        selected = (p < padj) & (np.abs(lfc) > lfc_min)
        return selected if gene_mask is None else selected & gene_mask

    return mask("real", DE_REAL_PADJ, DE_LFC_MIN), {
        t: mask("synthetic", t, DE_SYN_LFC_MIN[t]) for t in DE_SYN_PADJ}


def de_label(threshold):
    """Metric suffix preserving thousandths: 0.05 -> padj_005, 0.075 -> padj_0075."""
    return "padj_0" + f"{threshold:.3f}"[2:].removesuffix("0")


def de_metric_rows(artifacts, contrasts, gene_mask=None):
    """Return metrics.csv DE rows and the log2FC Pearson r of each full-cohort contrast.

    `contrasts` is [(column, case, reference)]. Names include the column only when a
    (case, reference) pair repeats. DE uses the full cohorts once: one split, zero spread.
    The slope is the least-squares fit of synthetic on real log2FC over real DE genes.
    """
    from collections import Counter
    import numpy as np

    pair_counts = Counter(contrast[1:] for contrast in contrasts)
    rows, lfc_r = [], []
    for index, (column, case, reference) in enumerate(contrasts):
        repeated = pair_counts[(case, reference)] > 1
        name = f"DE_{column + '_' if repeated else ''}{case}_vs_{reference}"
        real_sig, synthetic_sig = de_sets(artifacts, index, gene_mask)
        x = artifacts[f"full_c{index}_real_lfc"][real_sig]
        y = artifacts[f"full_c{index}_synthetic_lfc"][real_sig]
        x, y = x[np.isfinite(y)], y[np.isfinite(y)]
        r = (np.corrcoef(x, y)[0, 1] if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0
             else np.nan)
        slope = np.polyfit(x, y, 1)[0] if len(x) > 2 and np.std(x) > 0 else np.nan
        lfc_r.append(r)
        n_real = np.count_nonzero(real_sig)
        values = {"lfc_pearson": r, "lfc_slope": slope, "real_de_genes": n_real}
        for threshold, synthetic_mask in synthetic_sig.items():
            label = de_label(threshold)
            overlap = np.count_nonzero(real_sig & synthetic_mask)
            total = np.count_nonzero(synthetic_mask)
            values |= {f"synthetic_de_genes_{label}": total,
                       f"overlap_de_genes_{label}": overlap,
                       f"overlap_fraction_{label}": overlap / n_real if n_real else np.nan,
                       f"precision_{label}": overlap / total if total else np.nan}
        rows.extend(("DE", f"{name}_{key}", "mean", value, 0.0, 1)
                    for key, value in values.items())
    return rows, lfc_r


def load_gene_counts(config_path, artifacts):
    """Read reporting counts, rejecting settings that differ from saved metrics."""
    import yaml

    counts = {key: int(artifacts[key]) for key in
              ("n_genes_aux", "n_genes_secret", "n_genes_de") if key in artifacts}
    if config_path is not None:
        with config_path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        for key in ("n_genes_aux", "n_genes_secret", "n_genes_de"):
            if key not in counts or counts[key] != config[key]:
                raise ValueError(f"Saved {key} differs from the benchmark config; rerun compute_metrics.py")
            counts[key] = int(config[key])
    if "n_genes_de" not in counts:
        counts["n_genes_de"] = max(len(artifacts[key]) for key in artifacts
                                    if key.endswith("_de_genes"))
    return counts
