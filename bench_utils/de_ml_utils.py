import csv
import gzip
import io
import re
from functools import lru_cache
from importlib import resources

import numpy as np
from scipy.stats import spearmanr

"""
Correlation computer for differential expression results between real and synthetic data.
"""

def compute_de_corrs(de_real, de_syn, clip_max=30, mod_genes=True,
                     return_fold_corrs=False):
    """Correlate aligned DE results over real minimum-DE genes by default.

    Return mean and standard deviation of fold-wise Pearson correlations for
    log2 fold change and clipped -log10(adjusted p-value). Optionally include
    the fold-wise Pearson and Spearman values for plots.
    """
    lfc_corrs, pval_corrs = [], []
    lfc_spearman, pval_spearman = [], []
    for real, syn in zip(de_real, de_syn):
        aligned = align_de_to_real(real, syn)
        genes = aligned["genes_aligned"]
        real_index = {gene: idx for idx, gene in enumerate(real["genes_all"])}
        indices = [real_index[gene] for gene in genes]

        lfc_real = np.nan_to_num(np.asarray(real["lfc_all"])[indices], nan=0.0)
        lfc_syn = np.nan_to_num(aligned["lfc_aligned"], nan=0.0)
        padj_real = np.nan_to_num(np.asarray(real["padj_all"])[indices], nan=1.0)
        padj_syn = np.nan_to_num(aligned["padj_aligned"], nan=1.0)
        selected = padj_real < 0.05 if mod_genes else slice(None)
        with np.errstate(divide="ignore"):
            pval_real = np.clip(-np.log10(padj_real), 0, clip_max)
            pval_syn = np.clip(-np.log10(padj_syn), 0, clip_max)

        for x, y, pearson, spearman in ((lfc_real, lfc_syn, lfc_corrs, lfc_spearman),
                                        (pval_real, pval_syn, pval_corrs, pval_spearman)):
            x, y = x[selected], y[selected]
            valid = np.isfinite(x) & np.isfinite(y)
            x, y = x[valid], y[valid]
            defined = len(x) > 1 and np.std(x) > 0 and np.std(y) > 0
            pearson.append(np.corrcoef(x, y)[0, 1] if defined else np.nan)
            if return_fold_corrs:
                spearman.append(spearmanr(x, y).statistic if defined else np.nan)

    summary = (np.mean(lfc_corrs), np.mean(pval_corrs),
               np.std(lfc_corrs), np.std(pval_corrs))
    if return_fold_corrs:
        return (*summary, lfc_corrs, pval_corrs, lfc_spearman, pval_spearman)
    return summary


def align_de_to_real(de_real, de_syn):
    """
    Align real and synthetic DE results to the same set of genes."""

    genes_real = np.array(de_real["genes_all"])
    genes_syn = np.array(de_syn["genes_all"])
    lfc_syn = np.array(de_syn["lfc_all"])
    pval_syn = np.array(de_syn["padj_all"])

    syn_gene_to_idx = {gene: idx for idx, gene in enumerate(genes_syn)}

    aligned_genes = []
    aligned_lfc_syn = []
    aligned_pval_syn = []

    for gene in genes_real:
        if gene in syn_gene_to_idx:
            syn_idx = syn_gene_to_idx[gene]
            aligned_genes.append(gene)           # ← only append when found
            aligned_lfc_syn.append(lfc_syn[syn_idx])
            aligned_pval_syn.append(pval_syn[syn_idx])

    return {
        "lfc_aligned": np.array(aligned_lfc_syn),
        "padj_aligned": np.array(aligned_pval_syn),
        "genes_aligned": np.array(aligned_genes),  # ← common genes only
    }


def compute_shap_correlation(shap0_list, shap1_list, sp_k=150, r_k=150):
    """
    Compute Spearman and Pearson correlations of mean |SHAP| values between real and synthetic.

    For binary, computes a single correlation per fold.
    For multiclass, computes per-class correlations per fold and returns
    a (n_classes,) array of median correlations across folds.

    Parameters
    ----------
    shap0_list : list of arrays
        List of 5 real SHAP arrays. Each array is (n_samples, n_features) for binary
        or (n_samples, n_features, n_classes) for multiclass
    shap1_list : list of arrays
        List of 5 synthetic SHAP arrays matching shap0_list structure
    sp_k : int
        Number of top features (by real mean |SHAP|) to use for Spearman correlation
    r_k : int
        Number of top features (by real mean |SHAP|) to use for Pearson correlation

    Returns
    -------
    dict
        - "median_sp_rho": scalar (binary) or (n_classes,) array of median Spearman correlations
        - "median_r": scalar (binary) or (n_classes,) array of median Pearson correlations  
        - "std_sp_rho": scalar (binary) or (n_classes,) array of std across folds for Spearman
        - "std_r": scalar (binary) or (n_classes,) array of std across folds for Pearson
        - "raw_correlations_sp": list of scalars (binary) or list of (n_classes,) arrays for Spearman
        - "raw_correlations_r": list of scalars (binary) or list of (n_classes,) arrays for Pearson
    """
    shap0_probe = np.asarray(shap0_list[0])
    is_multiclass = shap0_probe.ndim == 3 and shap0_probe.shape[2] > 1

    if is_multiclass:
        n_classes = shap0_probe.shape[2]
        # Each element: (n_classes,) array of per-class correlations
        fold_corrs_sp = []
        fold_corrs_r = []

        for shap0, shap1 in zip(shap0_list, shap1_list):
            shap0_arr = np.asarray(shap0)
            shap1_arr = np.asarray(shap1)
            class_corrs_sp = np.empty(n_classes)
            class_corrs_r = np.empty(n_classes)
            for c in range(n_classes):
                mas0_c = np.abs(shap0_arr[:, :, c]).mean(axis=0)
                mas1_c = np.abs(shap1_arr[:, :, c]).mean(axis=0)
                
                # Spearman correlation
                ix0_sp = np.argsort(mas0_c)[::-1][:sp_k]
                if np.count_nonzero(mas1_c[ix0_sp]) == 0:
                    print("WARNING: model shap vals are all zeros for Spearman")
                    if np.count_nonzero(mas0_c[ix0_sp]) == 0:
                        print(f"WARNING: class {c} — real SHAP all zeros, skipping")
                        class_corrs_sp[c] = np.nan  # not 0
                    else:
                        class_corrs_sp[c] = 0
                else:
                    class_corrs_sp[c] = spearmanr(mas0_c[ix0_sp], mas1_c[ix0_sp])[0]
                
                # Pearson correlation
                ix0_r = np.argsort(mas0_c)[::-1][:r_k]
                if np.count_nonzero(mas1_c[ix0_r]) == 0:
                    print("WARNING: model shap vals are all zeros for Pearson")
                    if np.count_nonzero(mas0_c[ix0_r]) == 0:
                        print(f"WARNING: class {c} — real SHAP all zeros, skipping")
                        class_corrs_r[c] = np.nan  # not 0
                    else:
                        class_corrs_r[c] = 0
                else:
                    class_corrs_r[c] = np.corrcoef(mas0_c[ix0_r], mas1_c[ix0_r])[0, 1]
            
            fold_corrs_sp.append(class_corrs_sp)
            fold_corrs_r.append(class_corrs_r)

        # (n_folds, n_classes)
        stacked_sp = np.vstack(fold_corrs_sp)
        stacked_r = np.vstack(fold_corrs_r)

        return {
            "median_sp_rho": np.nanmedian(stacked_sp, axis=0),   # (n_classes,)
            "median_r": np.nanmedian(stacked_r, axis=0),        # (n_classes,)
            "std_sp_rho": np.nanstd(stacked_sp, axis=0),        # (n_classes,)
            "std_r": np.nanstd(stacked_r, axis=0),              # (n_classes,)
            "raw_correlations_sp": fold_corrs_sp,                # list of (n_classes,) arrays
            "raw_correlations_r": fold_corrs_r,                  # list of (n_classes,) arrays
        }
    else:
        correlations_sp = []
        correlations_r = []
        for shap0, shap1 in zip(shap0_list, shap1_list):
            shap0_arr = np.asarray(shap0)
            shap1_arr = np.asarray(shap1)
            mas0 = np.abs(shap0_arr).mean(axis=0)
            mas1 = np.abs(shap1_arr).mean(axis=0)
            
            # Spearman correlation
            ix0_sp = np.argsort(mas0)[::-1][:sp_k]
            if np.count_nonzero(mas1) == 0:
                print("WARNING: model shap vals are all zeros for Spearman")
                corr_sp = 0
            else:
                corr_sp = spearmanr(mas0[ix0_sp], mas1[ix0_sp])[0]
            correlations_sp.append(corr_sp)
            
            # Pearson correlation
            ix0_r = np.argsort(mas0)[::-1][:r_k]
            if np.count_nonzero(mas1) == 0:
                print("WARNING: model shap vals are all zeros for Pearson")
                corr_r = 0
            else:
                corr_r = np.corrcoef(mas0[ix0_r], mas1[ix0_r])[0, 1]
            correlations_r.append(corr_r)

        return {
            "median_sp_rho": np.nanmedian(correlations_sp),
            "median_r": np.nanmedian(correlations_r),
            "std_sp_rho": np.nanstd(correlations_sp),
            "std_r": np.nanstd(correlations_r),
            "raw_correlations_sp": correlations_sp,
            "raw_correlations_r": correlations_r,
        }


@lru_cache(maxsize=1)
def _ensembl_symbol_map():
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


def map_ensembl_to_symbols(ensembl_ids, dataset_name=None):
    """
    Takes a list of Ensembl IDs, returns a list of gene symbols in the same order.
    Unmapped IDs are kept as-is. Strips version suffixes (e.g., .14) for lookup.

    Uses the packaged HGNC mapping and does not require network access.

    Parameters
    ----------
    ensembl_ids : list
        List of Ensembl IDs to map.
    dataset_name : str, optional
        Retained for backward compatibility; no longer used.

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
