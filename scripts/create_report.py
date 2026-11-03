#!/usr/bin/env python3
"""Prepare benchmark report data and render the report with Typst."""

import argparse
from collections import Counter
from datetime import date
from importlib import metadata
import json
import os
import re
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import typst

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from bench_utils.config import (
    DE_SYN_PADJ, de_label, de_sets, load_gene_counts)
from bench_utils.de_ml_utils import map_ensembl_to_symbols

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output_dir", type=Path)
parser.add_argument("input_dir", type=Path)
parser.add_argument("--manifest", type=Path)
parser.add_argument("--bench-config", type=Path)
parser.add_argument("--cohort-name", default=None)
args = parser.parse_args()
dataset_dir = args.input_dir.resolve()
dataset_name = (dataset_dir.parent.name if dataset_dir.name == "real_data"
                else dataset_dir.name)

artifact_dir = args.output_dir / "artifacts"
metrics = pd.read_csv(artifact_dir / "metrics.csv")
clinical = pd.read_csv(args.input_dir / "metadata.csv", index_col=0)
expression = pd.read_csv(args.input_dir / "expression.csv.gz", index_col=0)
manifest = pd.read_csv(
    args.manifest or args.input_dir / "split_manifest.csv", dtype={"sample_id": str}
)
fig_dir = artifact_dir / "figs"
fig_dir.mkdir(parents=True, exist_ok=True)
template = Path(__file__).parent / "report" / "template" / "report.typ"

try:
    report_paths = (template.resolve(), args.output_dir.resolve())
    typst_root = Path(os.path.commonpath(report_paths))
except ValueError as error:
    message = "The report template and output directory must be on the same drive."
    raise SystemExit(message) from error



def _status(name, value):
    """Return Pass/Fail against hard-coded report thresholds, or N/A when untested."""
    trtr = metrics.loc[metrics["metric"] == "roc_auc_trtr", "value"].iloc[0]
    checks = {
        "coexpression_spearman": value >= 0.8,
        "umap_mmd": value <= 0.1,
        "roc_auc_tstr": value >= trtr - 0.05,
        "lfc_pearson": value >= 0.8,
        f"overlap_fraction_{de_label(DE_SYN_PADJ[-1])}": value >= 0.8,
        "shap_pearson": value >= 0.8,
    }
    key = next((key for key in checks if name.endswith(key)), None)
    if key is None or not np.isfinite(value):
        return "N/A"
    return "Pass" if checks[key] else "Fail"


def _metric(name, optional=False, decimals=3):
    rows = metrics.loc[metrics["metric"] == name]
    if rows.empty:
        if optional:
            return {"estimate": "N/A", "value": "N/A", "std": "N/A", "splits": "N/A",
                    "status": "N/A"}
        raise ValueError(f"Missing metric {name}; rerun compute_metrics.py")
    row = rows.iloc[0]
    estimate = "N/A" if not np.isfinite(row["value"]) else (
        (f'{row["value"]:.{decimals}f}' if row["n_splits"] == 1 or not np.isfinite(row["std"])
         else f'{row["value"]:.{decimals}f} ± {row["std"]:.{decimals}f}')
    )
    return {
        "estimate": estimate,
        "value": f'{row["value"]:.{decimals}f}',
        "std": f'{row["std"]:.{decimals}f}' if np.isfinite(row["std"]) else "N/A",
        "splits": str(int(row["n_splits"])),
        "status": _status(name, row["value"]),
    }


def _template_path(path):
    """Return an absolute Typst path within the compilation root."""
    return "/" + path.resolve().relative_to(typst_root).as_posix()


def _figure(name):
    path = fig_dir / name
    return _template_path(path) if path.is_file() else None


metric_names = (
    "roc_auc_trtr",
    "roc_auc_tstr",
    "coexpression_spearman",
    "umap_mmd",
    "privacy_p5_ratio",
    "shap_pearson",
    "shap_spearman",
)
with np.load(
    artifact_dir / "benchmark_artifacts.npz", allow_pickle=False
) as artifacts:
    target = str(artifacts["target"])
    contrasts = [tuple(map(str, pair)) for pair in artifacts["contrasts"]]
    contrast_columns = (
        artifacts["contrast_columns"].tolist() if "contrast_columns" in artifacts
        else [target] * len(contrasts)
    )
    # Expression filter applied by compute_metrics.py to the reported DE summaries.
    gene_mask = artifacts["full_de_expression_mask"]
    anchor_cols = artifacts["anchor_cols"].tolist()
    qi_cols = (artifacts["qi_cols"].tolist() if "qi_cols" in artifacts
               else [column for column in anchor_cols if column != target])
    clinical_secrets = (
        artifacts["clinical_secrets"].tolist() if "clinical_secrets" in artifacts
        else [column for column in clinical
              if column not in qi_cols and "id" not in column.lower()]
    )
    class_labels = list(dict.fromkeys(map(str, artifacts["class_labels"])))
    shap_multiclass = artifacts["f1_real_shap"].ndim == 3
    if not all(key in artifacts for key in (
        "shap_mean_r", "shap_mean_sp_rho", "shap_raw_correlations_r",
        "shap_raw_correlations_sp", "shap_signal_failures")):
        raise ValueError("Saved SHAP metrics are outdated; rerun compute_metrics.py")
    gene_counts = load_gene_counts(args.bench_config, artifacts)
    if gene_counts["n_genes_de"] == -1:
        gene_counts["n_genes_de"] = len(artifacts["full_de_genes"])
    n_genes_aux, n_genes_secret = gene_counts["n_genes_aux"], gene_counts["n_genes_secret"]
    splits = artifacts["splits"].tolist()
    valid_shap = (
        np.isfinite(artifacts["shap_raw_correlations_r"]).reshape(len(splits), -1).all(axis=1)
        & np.isfinite(artifacts["shap_raw_correlations_sp"]).reshape(len(splits), -1).all(axis=1))
    failed_folds = int(np.count_nonzero(
        artifacts["shap_signal_failures"].reshape(len(splits), -1).any(axis=1)))
    metadata_path = artifact_dir / "figure_metadata.json"
    figure_metadata = (json.loads(metadata_path.read_text(encoding="utf-8"))
                       if metadata_path.exists() else {})
    figure_split = figure_metadata.get("figure_split", splits[0])
    # create_figs.py also requires non-zero plotted SHAP values, so trust its choice.
    shap_split = figure_metadata.get("shap_split", next(
        (split for split, valid in zip(splits, valid_shap) if valid), None))
    # Per-gene DE results, one CSV per contrast; DE flags use the reported (filtered) genes.
    de_genes = artifacts["full_de_genes"]
    de_dir = artifact_dir / "de"
    de_dir.mkdir(exist_ok=True)
    symbols = map_ensembl_to_symbols(de_genes.tolist())
    de_files = []
    for index, ((case, reference), column) in enumerate(zip(contrasts, contrast_columns)):
        real_de, synthetic_de = de_sets(artifacts, index, gene_mask)
        path = de_dir / re.sub(r"[^\w.-]+", "_", f"de_{column}_{case}_vs_{reference}.csv")
        pd.DataFrame({
            "gene": de_genes, "gene_symbol": symbols,
            **{f"{kind}_{name}": artifacts[f"full_c{index}_{kind}_{key}"]
               for kind in ("real", "synthetic")
               for name, key in (("log2fc", "lfc"), ("padj", "padj"))},
            "passes_expression_filter": gene_mask, "real_de": real_de,
            **{f"synthetic_de_{rule}": mask
               for rule, mask in zip(("strict", "relaxed"), synthetic_de.values())},
        }).to_csv(path, index=False)
        de_files.append(path.relative_to(args.output_dir).as_posix())
    # Genes shared by the real- and synthetic-trained top 10 mean |SHAP|, per class and split.
    shap_labels = class_labels if shap_multiclass else class_labels[1:2]
    top10 = {label: [] for label in shap_labels}
    for fold in range(1, len(splits) + 1):
        real, synthetic = (
            np.nanmean(np.abs(artifacts[f"f{fold}_{kind}_shap"]), axis=0)
            .reshape(-1, len(shap_labels)) for kind in ("real", "synthetic"))
        for c, label in enumerate(shap_labels):
            top10[label].append(len(set(np.argsort(real[:, c])[::-1][:10])
                                    & set(np.argsort(synthetic[:, c])[::-1][:10])))



partition_sizes = manifest.groupby(["split", "partition"]).size()
train_sizes = partition_sizes.xs("train", level="partition").to_numpy()
test_sizes = partition_sizes.xs("test", level="partition").to_numpy()
synthetic_targets = [
    pd.read_csv(
        artifact_dir / f"syn_{dataset_name}_fold_{fold}.csv", usecols=[target]
    )[target]
    for fold in range(1, manifest["split"].nunique() + 1)
]


def _range(values):
    low, high = int(np.min(values)), int(np.max(values))
    return str(low) if low == high else f"{low}-{high}"


def _percent(name):
    """Metric entry with the estimate shown as a whole-number percentage."""
    entry = _metric(name)
    value = metrics.loc[metrics["metric"] == name, "value"].iloc[0]
    return entry | {"estimate": f"{value:.0%}" if np.isfinite(value) else "N/A"}


de_results = []
pair_counts = Counter(contrasts)
for index, ((case, reference), column) in enumerate(zip(contrasts, contrast_columns)):
    repeated = pair_counts[(case, reference)] > 1
    name = f"{column}_" if repeated else ""
    prefix = f"DE_{name}{case}_vs_{reference}"
    contrast_label = f"{case} vs {reference}"
    if column != target or repeated:
        contrast_label = f"{column}: {contrast_label}"
    synthetic_thresholds = [{
        "threshold": f"{threshold:g}",
        "total": _metric(f"{prefix}_synthetic_de_genes_{label}", decimals=0)["value"],
        "recovered": _metric(f"{prefix}_overlap_de_genes_{label}", decimals=0)["value"],
        "recovery_percent": _percent(f"{prefix}_overlap_fraction_{label}")["estimate"],
        "precision": _percent(f"{prefix}_precision_{label}")["estimate"],
    } for threshold, label in ((t, de_label(t)) for t in DE_SYN_PADJ)]
    de_results.append({
        "contrast": contrast_label,
        "case": case,
        "lfc": _metric(f"{prefix}_lfc_pearson"),
        "slope": _metric(f"{prefix}_lfc_slope", decimals=2),
        "real_de": _metric(f"{prefix}_real_de_genes", decimals=0),
        "synthetic_thresholds": synthetic_thresholds,
        "recovery": _percent(f"{prefix}_overlap_fraction_{de_label(DE_SYN_PADJ[-1])}"),
    })

shap_results = []
for label in class_labels:
    pearson = f"shap_pearson [{label}]"
    spearman = f"shap_spearman [{label}]"
    if pearson in set(metrics["metric"]) and spearman in set(metrics["metric"]):
        shap_results.append({
            "class": label,
            "pearson": _metric(pearson),
            "spearman": _metric(spearman),
            "top10": f"{np.mean(top10[label]):.1f}/10" if label in top10 else "N/A",
        })

privacy_metrics = {
    f"attribute_inference_{attack}{suffix}": _metric(
        f"attribute_inference_{attack}{suffix}", decimals=decimals)
    for attack in ("standard", "clinical_qi")
    for suffix, decimals in (("", 3), ("_columns_gt_02", 0))}
attribute = pd.read_csv(artifact_dir / "attribute_inference.csv")
attribute["excess"] = (
    attribute["training_attack_success_rate"]
    - attribute["control_attack_success_rate"])

# Absolute success rates behind the five largest per-secret mean excesses of each attack.
rates = ("training_attack_success_rate", "control_attack_success_rate", "excess")
top_secrets = []
for attack, label in (("standard", f"Clinical QI + {n_genes_aux} genes"),
                      ("clinical_qi", "Clinical QI only")):
    rows = attribute.loc[(attribute["attack"] == attack) & np.isfinite(attribute["excess"])]
    for secret, row in rows.groupby("secret")[list(rates)].mean().nlargest(5, "excess").iterrows():
        symbol = map_ensembl_to_symbols([secret])[0]
        top_secrets.append({
            "attack": label,
            "secret": (f"{symbol} (gene)" if secret in expression.columns else
                       secret.removeprefix("Sample Characteristic[").removesuffix("]")),
            **{rate.split("_")[0]: f"{row[rate]:.2f}" for rate in rates}})

# Provenance: Anonymeter lives in its own environment (bench_utils/attr_infer.py).
anonymeter_paths = [str(path) for pattern in ("*/site-packages", "*/*/site-packages")
                    for path in (root / ".venv_anonymeter").glob(pattern)]
versions = {name: metadata.version(package) for name, package in (
    ("PyDESeq2", "pydeseq2"), ("scikit-learn", "scikit-learn"), ("UMAP", "umap-learn"),
    ("SHAP", "shap"))}
versions["Anonymeter"] = next(
    (dist.version for dist in metadata.distributions(path=anonymeter_paths)
     if dist.metadata["Name"].lower() == "anonymeter"), "not installed")
pipeline = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]

# Data dictionary (Table 7: release layout, gene IDs) is fixed per release, so it is stored
# once in artifacts/data_files.json rather than recomputed here.
data_files = json.loads((artifact_dir / "data_files.json").read_text(encoding="utf-8"))
data_files["de_files"] = de_files

report_data = {
    "metrics": {name: _metric(name) for name in metric_names} | privacy_metrics,
    "cohort": {
        "name": args.cohort_name or args.input_dir.name,
        "ml_target": target,
        "real_samples": str(len(expression)),
        "genes": str(len(expression.columns)),
        "clinical_vars": str(len(clinical.columns)),
        "splits": str(len(train_sizes)),
        "train_samples": _range(train_sizes),
        "test_samples": _range(test_sizes),
        "synthetic_samples": _range([len(values) for values in synthetic_targets]),
        "classes": ", ".join(map(str, clinical[target].dropna().unique())),
        "n_genes_de": f'{gene_counts["n_genes_de"]:,}',
        "de_genes_tested": f"{len(de_genes):,}",
        "de_genes_reported": f"{int(gene_mask.sum()):,}",
        "de_all_genes": len(de_genes) == len(expression.columns),
        "n_genes_aux": str(n_genes_aux),
        "n_genes_secret": str(n_genes_secret),
        "anchor_cols": ", ".join(anchor_cols),
        "qi_cols": ", ".join(qi_cols) or "none",
        "clinical_secrets": ", ".join(clinical_secrets) or "none",
    },
    "anchor_cols": anchor_cols,
    "figure_columns": {
        _template_path(fig_dir / f"{kind}_{index}.pdf"): column
        for kind in ("de", "gene_counts")
        for index, column in enumerate(contrast_columns)
    },
    "de_contrasts": [
        {"column": column, "categories": [case, reference]}
        for (case, reference), column in zip(contrasts, contrast_columns)
    ],
    "de_results": de_results,
    "shap_results": shap_results,
    "top_secrets": top_secrets,
    "provenance": {
        "date": date.today().isoformat(),
        "pipeline": f'{pipeline["name"]} {pipeline["version"]}',
        "packages": ", ".join(f"{name} {version}" for name, version in versions.items()),
    },
    "data_files": data_files,
    "figure_split": str(figure_split),
    "shap_split": shap_split,
    "shap_failed_folds": failed_folds,
    "shap_total_folds": len(splits),
    "figures": {
        "expression_fidelity": _figure("expression_fidelity.pdf"),
        "clinical_fidelity": [
            {"variable": column, "path": _figure(f"clinical_fidelity{index}.pdf")}
            for index, column in enumerate(
                column for column in anchor_cols if column != target)
            if (fig_dir / f"clinical_fidelity{index}.pdf").is_file()][:2],
        "gene_counts": [
            _template_path(path) for path in (
                fig_dir / f"gene_counts_{index}.pdf" for index in range(len(contrasts))
            ) if path.is_file()],
        "differential_expression": [
            _template_path(path) for path in (
                fig_dir / f"de_{index}.pdf" for index in range(len(contrasts))
            ) if path.is_file()
        ],
        "shap": [] if shap_split is None else [
            _template_path(path) for path in (
                (fig_dir / f"SHAP_Synthetic_{index}.pdf"
                 for index in range(len(class_labels)))
                if shap_multiclass else (fig_dir / "SHAP_Synthetic.pdf",)
            ) if path.is_file()
        ],
        "privacy": _figure("privacy.pdf"),
        "auc_folds": _figure("auc_folds.pdf"),
        "gene_stats": _figure("gene_stats.pdf"),
    },
}

data_path = artifact_dir / "report_data.json"
data_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")


# Adjust the privacy presentation without changing the shared template file.
report_source = template.read_text(encoding="utf-8").replace(
    '..de-results.enumerate().map(((i, item)) => (',
    '..de-results.enumerate().map(((i, item)) => ('
    '\n      ..(if i > 0 { (table.hline(stroke: rule),) } else { () }),')
for old, new in (
    ('Adjusted p-values and log2 fold-changes were aligned by gene.\n',
     'Adjusted p-values and log2 fold-changes were aligned by gene.\n'
     'DE was run on all subjects, but only the top 80% of genes with the highest mean '
     'CPM in the real cohort (#(cohort.de_genes_reported) of #(cohort.de_genes_tested) '
     'tested genes) were considered in the figures and tables.\n'),
    ('"styles.typ"', json.dumps(_template_path(template.with_name("styles.typ")))),
    ('#for panel in figures.clinical_fidelity {\n  figure(\n    image(panel.path, width: 75%),\n'
     '    caption: [*Clinical fidelity of the synthetic cohort.* Panels compare full real and dbTwin '
     'synthetic distributions of #raw(panel.variable), conditioned on ML-target class.],\n  )\n}',
     '#let panels = figures.clinical_fidelity.enumerate().map(((i, panel)) => (\n'
     '  label: "ABCDEFGH".at(i), ..panel))\n'
     '#if panels.len() > 0 {\n  figure(\n'
     '    stack(spacing: 8pt, ..panels.map(panel => grid(\n'
     '      columns: (auto, 1fr), align: (left + top, center + top),\n'
     '      [*#panel.label*], image(panel.path, width: 85%)))),\n'
     '    caption: [*Clinical fidelity of anchor columns within ML-target classes.* Full real '
     'versus full dbTwin synthetic distributions, one panel per target class: '
     '#panels.map(panel => [(#panel.label) #raw(panel.variable)]).join("; "). '
     'Continuous columns are violins with box plots (median, IQR, whiskers to 1.5 IQR); '
     'categorical columns are level prevalences, with missing values as their own level. Panel '
     'titles give the total variation distance (TVD; 0 = identical, 1 = no overlap), using 10 '
     'equal-width bins on the real class for continuous columns. Even a perfect generator gives '
     'non-zero TVD, so each title also gives the sampling noise: the 97.5th percentile of TVD '
     'between the real class and 500 same-size bootstrap resamples of itself. "within noise: yes" '
     'means the observed TVD is no larger than that.],\n  )\n}'),
    ('Clinical anchor-column distributions were compared within ML-target classes\n',
     'Clinical anchor-column distributions were compared within ML-target classes\n'
     'using total variation distance (TVD)\n'),
    ('*How do I interpret the co-expression matrix?*',
     '*What does TVD mean in the clinical fidelity figure?* TVD is the fraction of probability '
     'mass that differs between real and synthetic distributions (0 = identical, 1 = disjoint). '
     'Numeric columns are binned using fixed-width bins defined on the real cohort. Small '
     'classes give non-zero TVD from sampling alone, so judge each value against the '
     'sampling-noise bound in its panel title.\n\n'
     '*How do I interpret the co-expression matrix?*'),
    (' and the median-distance ratio\nwas #metric("privacy_median_ratio").estimate', ''),
    ('    ("DCR median ratio", metric("privacy_median_ratio").estimate),\n', ''),
    ('mean ± SD across outer splits or attribute-inference evaluations.',
     'DCR: mean ± SD across splits; attribute inference: median of the top 10 column means across splits.'),
    ('"Attribute inference: expression + clinical"',
     '[#(cohort.n_genes_aux) expression genes + QI \\ median of top 10 means]'),
    ('"Attribute inference: clinical only"', '[Clinical QI \\ median of top 10 means]'),
    ('averaged across all available secret-by-split evaluations',
     'averaged across splits per secret column, then summarized as the median of the 10 highest column means for each attack'),
    ('Excess attribute-inference success aggregated across all evaluated secrets and splits.',
     'Median of the 10 highest per-column mean excess-success values across splits for each attack.'),
    ('"Training - control success"', '"Median of top 10 means"'),
    ('the fifth-percentile and median\nsynthetic-to-training distances',
     'the fifth-percentile\nsynthetic-to-training distances'),
    ('rate, averaged over available secret-by-split evaluations for each attack.',
     'rate, averaged across splits per secret column, then summarized as the median of the 10 highest column means for each attack.')):
    report_source = report_source.replace(old, new)
privacy_start = report_source.index('#table-caption(5,')
privacy_end = report_source.index('#if figures.privacy', privacy_start)
privacy_table = report_source[privacy_start:privacy_end].replace(
    '("Metric", "Estimate"),',
    '("Metric", "Estimate", [Features with mean \\ excess > 0.2]),').replace(
    'metric("privacy_p5_ratio").estimate),',
    'metric("privacy_p5_ratio").estimate, "--"),').replace(
    'column-widths: (auto, auto), width: auto',
    'column-widths: (auto, auto, auto), width: auto')
for attack in ("standard", "clinical_qi"):
    count_metric = f'metric("attribute_inference_{attack}_columns_gt_02").estimate'
    privacy_table = privacy_table.replace(
        f'metric("attribute_inference_{attack}").estimate),',
        f'metric("attribute_inference_{attack}").estimate, {count_metric}),')
report_source = (
    report_source[:privacy_start] + privacy_table + report_source[privacy_end:])

typst.compile(
    report_source.encode("utf-8"),
    output=str(args.output_dir / f"benchmark_report_{dataset_name}.pdf"),
    root=str(typst_root),
    sys_inputs={"data": _template_path(data_path)},
)
