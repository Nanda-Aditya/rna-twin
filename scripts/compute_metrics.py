#!/usr/bin/env python3
"""Compute full-cohort fidelity and split-based ML and privacy metrics."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Resolve helpers from this repository before editable installations.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_utils.attr_infer import compute_secret_risk
from bench_utils.clinical_fidelity import compute_clinical_fidelity
from bench_utils.config import de_metric_rows
from bench_utils.de_ml_utils import compute_shap_correlation
from bench_utils.expression import log_cpm, n_ml_features, top_variable_genes
from bench_utils.gexp_de import gexp_de
from bench_utils.ml_wrapper import train_trtr_tstr
from bench_utils.privacy_metrics import compute_dcr
from bench_utils.umap_coexp import (
    compute_coexpression_artifacts,
    compute_umap_artifacts,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("input_dir", type=Path)
parser.add_argument("output_dir", type=Path)
parser.add_argument("--ml-target", required=True)
parser.add_argument("--manifest", type=Path)
parser.add_argument("--full-synthetic", type=Path)
parser.add_argument("--anchor-cols", required=True)
parser.add_argument("--qi-cols", nargs="*", required=True)
parser.add_argument("--n-genes-aux", type=int, default=50)
parser.add_argument("--n-genes-secret", type=int, default=50)
parser.add_argument("--n-genes-de", type=int, default=6000,
                    help="Number of DE genes; -1 uses all genes.")
parser.add_argument("--contrast", nargs=3, action="append",
                    metavar=("COLUMN", "CASE", "REFERENCE"))
args = parser.parse_args()
dataset_dir = args.input_dir.resolve()
dataset_name = (dataset_dir.parent.name if dataset_dir.name == "real_data"
                else dataset_dir.name)
if min(args.n_genes_aux, args.n_genes_secret) < 0:
    parser.error("gene counts must be non-negative")
if args.n_genes_de < -1:
    parser.error("--n-genes-de must be -1 (all genes) or non-negative")
anchor_cols = [column.strip() for column in args.anchor_cols.split(",")]

clinical = pd.read_csv(args.input_dir / "metadata.csv", index_col=0)
expression = pd.read_csv(args.input_dir / "expression.csv.gz", index_col=0)
manifest = pd.read_csv(
    args.manifest or args.input_dir / "split_manifest.csv",
    dtype={"sample_id": str},
)
cv_results = pd.read_csv(args.output_dir / "cv_results.csv")
clinical.index = clinical.index.map(str)
expression.index = expression.index.map(str)
real = expression.join(clinical, how="inner")
full_syn = pd.read_csv(
    args.full_synthetic or args.output_dir.parent / f"syn_{dataset_name}.csv", index_col=0
)
genes = expression.columns.to_numpy()
contrasts = [tuple(contrast) for contrast in args.contrast or []]
if not contrasts:
    parser.error("at least one --contrast COLUMN CASE REFERENCE is required")
selected_params = cv_results.loc[
    cv_results["selected"], ["C", "l1_ratio"]
].drop_duplicates()
if len(selected_params) != 1:
    raise ValueError("Expected one selected C and l1_ratio; rerun cv_trn.py")
C, l1_ratio = selected_params.iloc[0]
clinical_qi = args.qi_cols
clinical_secrets = [
    column for column in clinical
    if column not in clinical_qi and "id" not in column.lower()
]
values = {
    "roc_auc_trtr": ("ML", []),
    "roc_auc_tstr": ("ML", []),
    "privacy_p5_ratio": ("Privacy", []),
    "privacy_median_ratio": ("Privacy", []),
    "umap_mmd": ("Global fidelity", []),
    "coexpression_spearman": ("Global fidelity", []),
}
real_shap, synthetic_shap = [], []
attribute_inference, dcr = [], []
splits = [str(split) for split, _ in manifest.groupby("split", sort=True)]
artifacts = {"splits": np.asarray(splits),
             "contrasts": np.asarray([contrast[1:] for contrast in contrasts]),
             "contrast_columns": np.asarray([contrast[0] for contrast in contrasts]),
             "target": np.asarray(args.ml_target),
             "anchor_cols": np.asarray(anchor_cols, dtype=str),
             "qi_cols": np.asarray(clinical_qi, dtype=str),
             "clinical_secrets": np.asarray(clinical_secrets, dtype=str),
             "n_genes_aux": np.asarray(args.n_genes_aux),
             "n_genes_secret": np.asarray(args.n_genes_secret),
             "n_genes_de": np.asarray(args.n_genes_de)}
d_st, d_tt = [], []

for fold, (split, rows) in enumerate(manifest.groupby("split", sort=True), start=1):
    partition = rows["partition"].str.lower()
    trn = real.loc[rows.loc[partition == "train", "sample_id"]]
    tst = real.loc[rows.loc[partition == "test", "sample_id"]]
    syn = pd.read_csv(args.output_dir / f"syn_{dataset_name}_fold_{fold}.csv", index_col=0)
    x_trn, x_tst, x_syn = (log_cpm(frame[genes]) for frame in (trn, tst, syn))
    ml_real, ml_syn, ml_genes = train_trtr_tstr(
        x_trn, trn[args.ml_target], x_syn, syn[args.ml_target], x_tst, tst[args.ml_target],
        n_ml_features(len(trn), len(genes)), C, l1_ratio,
    )
    # Both models share the sorted class order, which SHAP arrays and class_labels follow.
    class_labels = list(ml_real["classes"])
    if list(ml_syn["classes"]) != class_labels:
        raise ValueError(f"Split {split}: synthetic classes {list(ml_syn['classes'])} "
                         f"differ from real classes {class_labels}")
    values["roc_auc_trtr"][1].append(np.mean(ml_real["auc"]))
    values["roc_auc_tstr"][1].append(np.mean(ml_syn["auc"]))
    for kind, model, shap_values in (("real", ml_real, real_shap),
                                    ("synthetic", ml_syn, synthetic_shap)):
        shap_values.append(np.asarray(model["shap"]))
        artifacts[f"f{fold}_{kind}_shap"] = shap_values[-1]
        artifacts[f"f{fold}_{kind}_auc"] = np.asarray(np.mean(model["auc"]))
    artifacts[f"f{fold}_shap_genes"] = np.asarray(ml_genes, dtype=str)

    privacy = compute_dcr(x_trn, x_syn)
    fold_d_st = np.asarray(privacy["d_st"]).ravel()
    fold_d_tt = np.asarray(privacy["d_stst"]).ravel()
    d_st.append(fold_d_st)
    d_tt.append(fold_d_tt)
    dcr.extend(
        {"split": split, "comparison": "syn_trn", "sample_id": str(sample_id),
         "distance": distance}
        for sample_id, distance in zip(syn.index, fold_d_st)
    )
    dcr.extend(
        {"split": split, "comparison": "trn_trn", "sample_id": str(sample_id),
         "distance": distance}
        for sample_id, distance in zip(trn.index, fold_d_tt)
    )
    values["privacy_p5_ratio"][1].append(privacy["p5_ratio"])
    values["privacy_median_ratio"][1].append(
        np.median(fold_d_st) / np.median(fold_d_tt)
    )
    ranked_genes = top_variable_genes(
        x_syn, min(args.n_genes_aux + args.n_genes_secret, len(genes))
    ).tolist()
    attack_genes = ranked_genes[:args.n_genes_aux]
    secret_genes = ranked_genes[args.n_genes_aux:]
    # The attacker's genes are log2(CPM + 1); gene secrets keep their released raw counts.
    attack_trn, attack_syn, attack_tst = (
        frame.drop(columns=attack_genes).join(x[attack_genes])
        for frame, x in ((trn, x_trn), (syn, x_syn), (tst, x_tst))
    )
    for secret in [*clinical_secrets, *secret_genes]:
        for attack, gene_aux in (("standard", attack_genes),
                                 ("clinical_qi", [])):
            result = compute_secret_risk(
                attack_trn, attack_syn, attack_tst, gene_aux, clinical_qi, secret, 42 + fold
            ) if gene_aux or clinical_qi else None
            attribute_inference.append(
                {"split": split, "attack": attack, "secret": secret,
                 **(result or {})}
            )

phtp_cols = list(dict.fromkeys([args.ml_target, *anchor_cols]))
phtp0, phtp1 = real[phtp_cols], full_syn[phtp_cols]
clinical_metrics = compute_clinical_fidelity(phtp0, phtp1, args.ml_target, anchor_cols)
clinical_data = pd.concat(
    frame.assign(source=source) for source, frame in (("real", phtp0), ("synthetic", phtp1))
)

x_real, x_full_syn = log_cpm(real[genes]), log_cpm(full_syn[genes])
umap_real, umap_syn, mmd = compute_umap_artifacts(x_real, x_full_syn)
co_genes, co_real, co_syn, coexpression = compute_coexpression_artifacts(x_real, x_full_syn)
for kind, embedding, frame in (("real", umap_real, real), ("synthetic", umap_syn, full_syn)):
    artifacts[f"full_umap_{kind}"] = np.asarray(embedding, dtype=float)
    artifacts[f"full_umap_{kind}_target"] = np.asarray(frame[args.ml_target], dtype=str)
artifacts["full_umap_mmd"] = np.asarray(mmd)
artifacts["full_coexpression_genes"] = np.asarray(co_genes, dtype=str)
artifacts["full_coexpression_real"] = co_real
artifacts["full_coexpression_synthetic"] = co_syn
artifacts["full_coexpression_rho"] = np.asarray(coexpression)
values["umap_mmd"][1].append(mmd)
values["coexpression_spearman"][1].append(coexpression)

# DESeq2 fits raw counts; -1 tests all genes, otherwise the most variable real log2(CPM + 1) genes.
de_genes = genes if args.n_genes_de == -1 else top_variable_genes(x_real, args.n_genes_de)
artifacts["full_de_genes"] = np.asarray(de_genes, dtype=str)
# DE summaries keep the 80% of tested genes with the highest mean CPM in the real cohort.
mean_cpm = real[genes].div(real[genes].sum(axis=1), axis=0)[de_genes].mean() * 1e6
artifacts["full_de_expression_mask"] = (mean_cpm >= mean_cpm.quantile(0.20)).to_numpy()
for contrast_index, (column, case, reference) in enumerate(contrasts):
    for kind, frame in (("real", real), ("synthetic", full_syn)):
        result = gexp_de(frame[[*de_genes, column]], column, (case, reference)).reindex(de_genes)
        artifacts[f"full_c{contrast_index}_{kind}_lfc"] = result["log2FoldChange"].to_numpy()
        artifacts[f"full_c{contrast_index}_{kind}_padj"] = result["padj"].to_numpy()

rows = []
for metric, (section, metric_values) in values.items():
    array = np.asarray(metric_values, dtype=float)
    rows.append((section, metric, "mean", np.nanmean(array), np.nanstd(array), len(array)))

# Attribute inference excess is training minus control success, averaged per secret over
# splits; each attack reports the median of its 10 largest secret means and the count above 0.2.
attacks = pd.DataFrame(attribute_inference)
attacks["excess"] = (attacks["training_attack_success_rate"]
                     - attacks["control_attack_success_rate"])
for attack in ("standard", "clinical_qi"):
    scored = attacks.loc[(attacks["attack"] == attack) & np.isfinite(attacks["excess"])]
    secret_means = scored.groupby("secret")["excess"].mean()
    n_scored = scored["split"].nunique()
    rows += [("Privacy", f"attribute_inference_{attack}", "median_top10",
              secret_means.nlargest(10).median(), np.nan, n_scored),
             ("Privacy", f"attribute_inference_{attack}_columns_gt_02", "count",
              (secret_means > 0.2).sum() if n_scored else np.nan, np.nan, n_scored)]

shap =compute_shap_correlation(real_shap, synthetic_shap, sp_k=50, r_k=50)
for metric, suffix, raw in (("shap_pearson", "r", "raw_correlations_r"),
                            ("shap_spearman", "sp_rho", "raw_correlations_sp")):
    means, stds = np.atleast_1d(shap[f"mean_{suffix}"]), np.atleast_1d(shap[f"std_{suffix}"])
    valid_folds = np.isfinite(shap[raw]).reshape(len(real_shap), -1).any(axis=1)
    rows.append(("ML", metric, "mean",
                 np.nanmean(means) if np.isfinite(means).any() else np.nan,
                 np.nanmean(stds) if np.isfinite(stds).any() else np.nan,
                 np.count_nonzero(valid_folds)))
de_rows, de_lfc_r = de_metric_rows(artifacts, contrasts, artifacts["full_de_expression_mask"])
rows.extend(de_rows)
for contrast_index, r in enumerate(de_lfc_r):
    artifacts[f"full_c{contrast_index}_lfc_pearson"] = np.asarray(r)

args.output_dir.mkdir(parents=True, exist_ok=True)
metrics = pd.DataFrame(
    rows, columns=("section", "metric", "statistic", "value", "std", "n_splits")
)
metrics["class_label"] = ""
labels = class_labels if np.asarray(real_shap[0]).ndim == 3 else class_labels[1:2]
for metric, correlations, means, stds in (
    ("shap_pearson", shap["raw_correlations_r"], shap["mean_r"], shap["std_r"]),
    ("shap_spearman", shap["raw_correlations_sp"], shap["mean_sp_rho"], shap["std_sp_rho"]),
):
    fold_values = pd.DataFrame(correlations, columns=labels)
    for index, label in enumerate(labels):
        values = fold_values[label].dropna()
        metrics.loc[len(metrics)] = (
            "ML", f"{metric} [{label}]", "mean", np.atleast_1d(means)[index],
            np.atleast_1d(stds)[index], len(values), label,
        )
metrics.to_csv(args.output_dir / "metrics.csv", index=False)
pd.DataFrame(attribute_inference).to_csv(
    args.output_dir / "attribute_inference.csv", index=False
)
pd.DataFrame(dcr).to_csv(args.output_dir / "dcr.csv", index=False)
clinical_metrics.to_csv(args.output_dir / "clinical_fidelity_metrics.csv", index=False)
clinical_data.to_csv(args.output_dir / "clinical_fidelity_data.csv", index=False)
artifacts["class_labels"] = np.asarray(class_labels, dtype=str)
artifacts["dcr_syn_trn"] = np.concatenate(d_st)
artifacts["dcr_trn_trn"] = np.concatenate(d_tt)
for key in ("mean_r", "std_r", "mean_sp_rho", "std_sp_rho"):
    artifacts[f"shap_{key}"] = np.asarray(shap[key], dtype=float)
for key in ("raw_correlations_r", "raw_correlations_sp", "signal_failures"):
    artifacts[f"shap_{key}"] = np.asarray(shap[key])
np.savez_compressed(args.output_dir / "benchmark_artifacts.npz", **artifacts)
