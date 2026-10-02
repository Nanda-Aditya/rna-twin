#!/usr/bin/env python3
"""Compute benchmark metrics from real and synthetic split data."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from bench_utils.attr_infer import compute_secret_risk
from bench_utils.clinical_fidelity import compute_clinical_fidelity
from bench_utils.de_ml_utils import compute_shap_correlation
from bench_utils.gexp_de import de_computes
from bench_utils.ml_wrapper import tune_and_train
from bench_utils.privacy_metrics import (
    compute_expression_privacy_metrics,
)
from bench_utils.umap_coexp import (
    compute_coexpression_artifacts,
    compute_umap_artifacts,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("input_dir", type=Path)
parser.add_argument("output_dir", type=Path)
parser.add_argument("--target", required=True)
parser.add_argument("--context-cols", required=True)
parser.add_argument("--n-genes-aux", type=int, default=50)
parser.add_argument("--n-genes-secret", type=int, default=50)
parser.add_argument("--contrast", nargs=2, action="append", metavar=("CASE", "REFERENCE"))
args = parser.parse_args()
if args.n_genes_aux < 0 or args.n_genes_secret < 0:
    parser.error("gene counts must be non-negative")
context_cols = [column.strip() for column in args.context_cols.split(",")]

clinical = pd.read_csv(args.input_dir / "clinical.csv", index_col=0)
expression = pd.read_csv(args.input_dir / "expression.csv", index_col=0)
manifest = pd.read_csv(args.input_dir / "split_manifest.csv", dtype={"sample_id": str})
cv_results = pd.read_csv(args.output_dir / "cv_results.csv")
clinical.index = clinical.index.map(str)
expression.index = expression.index.map(str)
real = expression.join(clinical, how="inner")
genes = expression.columns.to_numpy()
contrasts = args.contrast or []
if not contrasts:
    labels = sorted(real[args.target].dropna().unique())
    if len(labels) != 2:
        raise ValueError("Explicit --contrast CASE REFERENCE required for multiclass DE")
    contrasts = [(str(labels[1]), str(labels[0]))]
contrasts = [tuple(pair) for pair in contrasts]
selected_params = cv_results.loc[
    cv_results["selected"], ["C", "l1_ratio"]
].drop_duplicates()
if len(selected_params) != 1:
    raise ValueError("Expected one selected C and l1_ratio; rerun cv_trn.py")
hyp_params = {
    "clf__C": selected_params.iloc[0]["C"],
    "clf__l1_ratio": selected_params.iloc[0]["l1_ratio"],
}
clinical_qi = [column for column in context_cols if column != args.target]
clinical_secrets = [args.target, *(
    column for column in clinical
    if column not in [args.target, *clinical_qi] and "id" not in column.lower()
)]
values = {
    "roc_auc_trtr": ("ML", []),
    "roc_auc_tstr": ("ML", []),
    "privacy_p5_ratio": ("Privacy", []),
    "privacy_median_ratio": ("Privacy", []),
    "umap_mmd": ("Privacy", []),
    "coexpression_spearman": ("Privacy", []),
    "attribute_inference_standard": ("Privacy", []),
    "attribute_inference_clinical_qi": ("Privacy", []),
}
real_shap, synthetic_shap = [], []
real_de = [[] for _ in contrasts]
synthetic_de = [[] for _ in contrasts]
attribute_inference, dcr = [], []
class_labels = None
splits = [str(split) for split, _ in manifest.groupby("split", sort=True)]
artifacts = {"splits": np.asarray(splits), "contrasts": np.asarray(contrasts),
             "target": np.asarray(args.target),
             "context_cols": np.asarray(context_cols, dtype=str),
             "n_genes_aux": np.asarray(args.n_genes_aux),
             "n_genes_secret": np.asarray(args.n_genes_secret)}
d_strn, d_stst = [], []
clinical_metrics, clinical_data = [], []

for fold, (split, rows) in enumerate(manifest.groupby("split", sort=True), start=1):
    partition = rows["partition"].str.lower()
    trn = real.loc[rows.loc[partition == "train", "sample_id"]]
    tst = real.loc[rows.loc[partition == "test", "sample_id"]]
    syn = pd.read_csv(args.output_dir / f"df_syn_{split}.csv", index_col=0)
    phtp_cols = list(dict.fromkeys([args.target, *context_cols]))
    phtp0, phtp1 = trn[phtp_cols].copy(), syn[phtp_cols].copy()
    clinical_metrics.append(
        compute_clinical_fidelity(phtp0, phtp1, args.target, context_cols)
        .assign(split=split)
    )
    for source, frame in (("real", phtp0), ("synthetic", phtp1)):
        clinical_data.append(frame.assign(split=split, source=source))
    n_features = max(1, min(270, int(len(trn) * 0.9), len(genes)))
    ml_real, ml_syn, _, _, ml_genes = tune_and_train(
        trn, tst, syn, syn, syn, args.target, genes, n_features,
        hyp_params=hyp_params, do_dbtwin=True,
    )
    values["roc_auc_trtr"][1].append(np.mean(ml_real["auc"]))
    values["roc_auc_tstr"][1].append(np.mean(ml_syn["auc"]))
    if class_labels is None:
        class_labels = list(ml_real["orig_classes_"])
    for kind, model, shap_values in (("real", ml_real, real_shap),
                                    ("synthetic", ml_syn, synthetic_shap)):
        shap_array = np.asarray(model["shap"])
        if shap_array.ndim == 3:
            model_labels = list(model["orig_classes_"])
            order = [model_labels.index(label) for label in class_labels]
            shap_array = shap_array[:, :, order]
        shap_values.append(shap_array)
        artifacts[f"f{fold}_{kind}_shap"] = shap_array
        artifacts[f"f{fold}_{kind}_auc"] = np.asarray(np.mean(model["auc"]))
    artifacts[f"f{fold}_shap_genes"] = np.asarray(ml_genes, dtype=str)

    privacy = compute_expression_privacy_metrics(trn, syn, tst, genes)
    fold_d_strn = np.asarray(privacy["d_strn"]).ravel()
    fold_d_stst = np.asarray(privacy["d_stst"]).ravel()
    d_strn.append(fold_d_strn)
    d_stst.append(fold_d_stst)
    dcr.extend(
        {"split": split, "sample_id": str(sample_id),
         "dcr_synthetic_train": train, "dcr_synthetic_test": test}
        for sample_id, train, test in zip(
            syn.index, fold_d_strn, fold_d_stst
        )
    )
    values["privacy_p5_ratio"][1].append(
        np.percentile(privacy["d_strn"], 5)
        / np.percentile(privacy["d_stst"], 5)
    )
    values["privacy_median_ratio"][1].append(
        np.median(privacy["d_strn"]) / np.median(privacy["d_stst"])
    )
    umap, mmd = compute_umap_artifacts(trn, {"dbt": syn}, genes, args.target)
    co_genes, co_real, co_syn, coexpression = compute_coexpression_artifacts(
        trn, {"dbt": syn}, genes
    )
    for key, array in umap["dbt"].items():
        dtype = str if key.endswith("target") else float
        artifacts[f"f{fold}_umap_{key}"] = np.asarray(array, dtype=dtype)
    artifacts[f"f{fold}_umap_mmd"] = np.asarray(mmd["dbt"])
    artifacts[f"f{fold}_coexpression_genes"] = np.asarray(co_genes, dtype=str)
    artifacts[f"f{fold}_coexpression_real"] = co_real
    artifacts[f"f{fold}_coexpression_synthetic"] = co_syn["dbt"]
    artifacts[f"f{fold}_coexpression_rho"] = np.asarray(coexpression["dbt"])
    values["umap_mmd"][1].append(mmd["dbt"])
    values["coexpression_spearman"][1].append(coexpression["dbt"])

    mad = syn[genes].sub(syn[genes].median()).abs().median()
    ranked_genes = mad.nlargest(
        min(args.n_genes_aux + args.n_genes_secret, len(genes))
    ).index.tolist()
    attack_genes = ranked_genes[:args.n_genes_aux]
    secret_genes = ranked_genes[args.n_genes_aux:]
    for secret in [*clinical_secrets, *secret_genes]:
        for attack, gene_aux in (("standard", attack_genes),
                                 ("clinical_qi", [])):
            result = compute_secret_risk(
                trn, syn, tst, gene_aux, clinical_qi, secret, 42 + fold
            ) if gene_aux or clinical_qi else None
            attribute_inference.append(
                {"split": split, "attack": attack, "secret": secret,
                 **(result or {})}
            )
            values[f"attribute_inference_{attack}"][1].append(
                result["training_attack_success_rate"]
                - result["control_attack_success_rate"] if result else np.nan
            )

    de_genes = trn[genes].sub(trn[genes].median()).abs().median().nlargest(6000).index
    artifacts[f"f{fold}_de_genes"] = np.asarray(de_genes, dtype=str)
    for contrast_index, contrast in enumerate(contrasts):
        for kind, frame, results in (("real", trn, real_de[contrast_index]),
                                     ("synthetic", syn,
                                      synthetic_de[contrast_index])):
            result = de_computes(frame[[*de_genes, args.target]], args.target, contrast)
            results.append(result)
            aligned = pd.DataFrame({"lfc": result["lfc_all"],
                                    "padj": result["padj_all"]}, index=result["genes_all"])
            aligned = aligned.reindex(de_genes)
            artifacts[f"f{fold}_c{contrast_index}_{kind}_lfc"] = aligned["lfc"].to_numpy()
            artifacts[f"f{fold}_c{contrast_index}_{kind}_padj"] = aligned["padj"].to_numpy()

rows = []
for metric, (section, metric_values) in values.items():
    array = np.asarray(metric_values, dtype=float)
    rows.append((section, metric, "mean", np.nanmean(array), np.nanstd(array), len(array)))

shap = compute_shap_correlation(real_shap, synthetic_shap, sp_k=50, r_k=50)
rows.extend((
    ("ML", "shap_pearson", "median", np.nanmean(shap["median_r"]),
     np.nanmean(shap["std_r"]), len(real_shap)),
    ("ML", "shap_spearman", "median", np.nanmean(shap["median_sp_rho"]),
     np.nanmean(shap["std_sp_rho"]), len(real_shap)),
))
for contrast_index, (contrast, real_results, synthetic_results) in enumerate(
    zip(contrasts, real_de, synthetic_de)):
    contrast_name = "_vs_".join(map(str, contrast))
    real_counts, synthetic_counts, overlap_counts, lfc_corrs = [], [], [], []
    for fold in range(1, len(real_results) + 1):
        prefix = f"f{fold}_c{contrast_index}_"
        real_padj = artifacts[f"{prefix}real_padj"]
        synthetic_padj = artifacts[f"{prefix}synthetic_padj"]
        real_lfc = artifacts[f"{prefix}real_lfc"]
        synthetic_lfc = artifacts[f"{prefix}synthetic_lfc"]
        real_sig, synthetic_sig = real_padj < 0.05, synthetic_padj < 0.10
        overlap = real_sig & synthetic_sig
        real_counts.append(np.count_nonzero(real_sig))
        synthetic_counts.append(np.count_nonzero(synthetic_sig))
        overlap_counts.append(np.count_nonzero(overlap))
        selected = overlap & np.isfinite(real_lfc) & np.isfinite(synthetic_lfc)
        x, y = real_lfc[selected], synthetic_lfc[selected]
        r = (np.corrcoef(x, y)[0, 1] if len(x) > 1 and
             np.std(x) > 0 and np.std(y) > 0 else np.nan)
        lfc_corrs.append(r)
        artifacts[f"{prefix}lfc_pearson"] = np.asarray(r)
    rows.append(("DE", f"DE_{contrast_name}_lfc_pearson", "mean",
                 np.nanmean(lfc_corrs), np.nanstd(lfc_corrs), len(lfc_corrs)))
    for name, fold_counts in (
        ("real_de_genes", real_counts),
        ("synthetic_de_genes", synthetic_counts),
        ("overlap_de_genes", overlap_counts),
    ):
        rows.append(("DE", f"DE_{contrast_name}_{name}", "mean",
                     np.mean(fold_counts), np.std(fold_counts), len(fold_counts)))

args.output_dir.mkdir(parents=True, exist_ok=True)
metrics = pd.DataFrame(
    rows, columns=("section", "metric", "statistic", "value", "std", "n_splits")
)
metrics["class_label"] = ""
labels = class_labels if np.asarray(real_shap[0]).ndim == 3 else class_labels[1:2]
for metric, correlations in (
    ("shap_pearson", shap["raw_correlations_r"]),
    ("shap_spearman", shap["raw_correlations_sp"]),
):
    fold_values = pd.DataFrame(correlations, columns=labels)
    for label in labels:
        values = fold_values[label].dropna()
        metrics.loc[len(metrics)] = (
            "ML", f"{metric} [{label}]", "mean", values.mean(),
            values.std(ddof=0), len(values), label,
        )
metrics.to_csv(args.output_dir / "metrics.csv", index=False)
pd.DataFrame(attribute_inference).to_csv(
    args.output_dir / "attribute_inference.csv", index=False
)
pd.DataFrame(dcr).to_csv(args.output_dir / "dcr.csv", index=False)
pd.concat(clinical_metrics, ignore_index=True).to_csv(
    args.output_dir / "clinical_fidelity_metrics.csv", index=False
)
pd.concat(clinical_data).to_csv(
    args.output_dir / "clinical_fidelity_data.csv", index=False
)
artifacts["class_labels"] = np.asarray(class_labels, dtype=str)
artifacts["dcr_synthetic_train"] = np.concatenate(d_strn)
artifacts["dcr_synthetic_test"] = np.concatenate(d_stst)
for key in ("median_r", "std_r", "median_sp_rho", "std_sp_rho"):
    artifacts[f"shap_{key}"] = np.asarray(shap[key], dtype=float)
np.savez_compressed(args.output_dir / "benchmark_artifacts.npz", **artifacts)
