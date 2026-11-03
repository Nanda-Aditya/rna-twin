"""
SHAP-importance correlations between real- and synthetic-trained models, and Ensembl-to-symbol
gene mapping.
"""

import csv
import gzip
import io
import re
from functools import lru_cache
from importlib import resources

import numpy as np
from scipy.stats import spearmanr


def compute_shap_correlation(shap0_list, shap1_list, sp_k=50, r_k=50):
    """Return mean +/- SD of valid fold correlations of mean absolute SHAP.

    `shap0_list`/`shap1_list` hold one real-/synthetic-trained SHAP array per fold, shaped
    (n_samples, n_features) or (n_samples, n_features, n_classes). Per fold and class,
    features are ranked by real mean |SHAP|; the Pearson r uses the top `r_k` features and
    the Spearman rho the top `sp_k`. Both selected vectors must be finite and nonconstant.
    Binary summaries are scalars; multiclass summaries have one entry per class. Raw
    correlations retain NaN for undefined folds. signal_failures marks variable real top-50
    importance (always 50, whatever `r_k`) with zero synthetic importance, separately from
    other reasons for an undefined correlation.
    """
    multiclass = np.asarray(shap0_list[0]).ndim == 3
    n_classes = np.asarray(shap0_list[0]).shape[2] if multiclass else 1
    correlations_r, correlations_sp, failures = [], [], []
    for shap0, shap1 in zip(shap0_list, shap1_list):
        real = np.nanmean(np.abs(shap0), axis=0).reshape(-1, n_classes)
        synthetic = np.nanmean(np.abs(shap1), axis=0).reshape(-1, n_classes)
        fold_r, fold_sp = np.full(n_classes, np.nan), np.full(n_classes, np.nan)
        fold_failures = np.zeros(n_classes, dtype=bool)
        for c in range(n_classes):
            ranked = np.argsort(np.where(np.isfinite(real[:, c]), real[:, c],
                                        -np.inf))[::-1]
            selected = ranked[:50]
            x, y = real[selected, c], synthetic[selected, c]
            fold_failures[c] = (np.all(np.isfinite(x)) and np.std(x) > 0
                                and np.all(y == 0))
            for k, fold_corr, spearman in ((r_k, fold_r, False),
                                           (sp_k, fold_sp, True)):
                selected = ranked[:k]
                x, y = real[selected, c], synthetic[selected, c]
                if (len(x) > 1 and np.all(np.isfinite(x))
                        and np.all(np.isfinite(y)) and np.std(x) > 0
                        and np.std(y) > 0):
                    fold_corr[c] = (spearmanr(x, y).statistic if spearman
                                    else np.corrcoef(x, y)[0, 1])
        correlations_r.append(fold_r)
        correlations_sp.append(fold_sp)
        failures.append(fold_failures)
    result = {}
    for suffix, correlations in (("r", correlations_r),
                                 ("sp_rho", correlations_sp)):
        array = np.asarray(correlations)
        means, stds = np.full(n_classes, np.nan), np.full(n_classes, np.nan)
        for c in range(n_classes):
            if np.isfinite(array[:, c]).any():
                means[c], stds[c] = np.nanmean(array[:, c]), np.nanstd(array[:, c])
        result[f"mean_{suffix}"] = means if multiclass else means[0]
        result[f"std_{suffix}"] = stds if multiclass else stds[0]
        key = "raw_correlations_sp" if suffix == "sp_rho" else "raw_correlations_r"
        result[key] = array if multiclass else array[:, 0]
    result["signal_failures"] = np.asarray(failures) if multiclass else np.asarray(failures)[:, 0]
    return result


@lru_cache(maxsize=1)
def _ensembl_symbol_map():
    """Unversioned Ensembl gene ID -> HGNC symbol, from the packaged HGNC table (offline)."""
    mapping_file = resources.files("bench_utils").joinpath(
        "data/hgnc_ensembl_symbols_2026-07-07.tsv.gz"
    )
    with mapping_file.open("rb") as raw:
        with gzip.GzipFile(fileobj=raw) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                return {
                    row["ensembl_gene_id"]: row["symbol"]
                    for row in csv.DictReader(text, delimiter="\t")
                }


def map_ensembl_to_symbols(ensembl_ids):
    """
    Takes a list of Ensembl IDs, returns a list of gene symbols in the same order.
    Unmapped IDs are kept as-is. Strips version suffixes (e.g., .14) for lookup.

    Uses the packaged HGNC mapping and does not require network access.

    Parameters
    ----------
    ensembl_ids : list
        List of Ensembl IDs to map.

    Returns
    -------
    list
        Gene symbols in same order as input (unmapped IDs kept as-is).
    """
    # Strip version suffixes (.XX) before lookup
    clean_ids = [re.sub(r"^(ENSG\d+)\.\d+$", r"\1", eid) for eid in ensembl_ids]
    ensembl_to_symbol = _ensembl_symbol_map()

    # Preserve order, fallback to original ID if unmapped
    return [
        ensembl_to_symbol.get(clean_id, orig_id)
        for clean_id, orig_id in zip(clean_ids, ensembl_ids)
    ]
