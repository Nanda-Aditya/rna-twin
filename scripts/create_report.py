#!/usr/bin/env python3
"""Prepare benchmark report data and render the report with Typst."""

import argparse
import json
import os
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bench_utils.plot_utils_ml import MODEL_COLORS

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output_dir", type=Path)
parser.add_argument("input_dir", type=Path)
parser.add_argument("--cohort-name", default=None)
args = parser.parse_args()

artifact_dir = args.output_dir / "artifacts"
metrics = pd.read_csv(artifact_dir / "metrics.csv")
clinical = pd.read_csv(args.input_dir / "clinical.csv", index_col=0)
expression = pd.read_csv(args.input_dir / "expression.csv", index_col=0)
manifest = pd.read_csv(
    args.input_dir / "split_manifest.csv", dtype={"sample_id": str}
)
fig_dir = artifact_dir / "figs"
fig_dir.mkdir(parents=True, exist_ok=True)
template = Path(__file__).parent / "report" / "template" / "report.typ"


def _metric(name, optional=False, decimals=3):
    rows = metrics.loc[metrics["metric"] == name]
    if rows.empty:
        if optional:
            return {"estimate": "N/A", "value": "N/A", "std": "N/A", "splits": "N/A"}
        raise ValueError(f"Missing metric {name}; rerun compute_metrics.py")
    row = rows.iloc[0]
    estimate = "N/A" if not np.isfinite(row["value"]) else (
        f'{row["value"]:.{decimals}f} +/- {row["std"]:.{decimals}f}'
    )
    return {
        "estimate": estimate,
        "value": f'{row["value"]:.{decimals}f}',
        "std": f'{row["std"]:.{decimals}f}',
        "splits": str(int(row["n_splits"])),
    }


def _template_path(path):
    """Return a Typst path relative to the template that will consume it."""
    return Path(os.path.relpath(path.resolve(), template.parent.resolve())).as_posix()


def _figure(name):
    path = fig_dir / name
    return _template_path(path) if path.is_file() else None


metric_names = (
    "roc_auc_trtr",
    "roc_auc_tstr",
    "coexpression_spearman",
    "umap_mmd",
    "privacy_p5_ratio",
    "privacy_median_ratio",
    "attribute_inference_standard",
    "attribute_inference_clinical_qi",
    "shap_pearson",
    "shap_spearman",
)
with np.load(
    artifact_dir / "benchmark_artifacts.npz", allow_pickle=False
) as artifacts:
    target = str(artifacts["target"])
    contrasts = [tuple(map(str, pair)) for pair in artifacts["contrasts"]]
    class_labels = list(dict.fromkeys(map(str, artifacts["class_labels"])))
    n_genes_aux = int(artifacts["n_genes_aux"]) if "n_genes_aux" in artifacts else 50
    n_genes_secret = (int(artifacts["n_genes_secret"])
                      if "n_genes_secret" in artifacts else 50)

partition_sizes = manifest.groupby(["split", "partition"]).size()
train_sizes = partition_sizes.xs("train", level="partition").to_numpy()
test_sizes = partition_sizes.xs("test", level="partition").to_numpy()
synthetic_targets = [
    pd.read_csv(
        artifact_dir / f"df_syn_{split}.csv", usecols=[target]
    )[target]
    for split in sorted(manifest["split"].astype(str).unique())
]


def _range(values):
    low, high = int(np.min(values)), int(np.max(values))
    return str(low) if low == high else f"{low}-{high}"


class_distribution = []
for label, count in clinical[target].value_counts().items():
    proportions = np.asarray([
        (values == label).mean() for values in synthetic_targets
    ])
    class_distribution.append({
        "label": str(label),
        "real_count": str(int(count)),
        "real_percent": f"{100 * count / len(clinical):.1f}%",
        "synthetic_percent": (
            f"{100 * proportions.mean():.1f}% +/- "
            f"{100 * proportions.std():.1f}%"
        ),
    })

de_results = []
for case, reference in contrasts:
    prefix = f"DE_{case}_vs_{reference}"
    de_results.append({
        "contrast": f"{case} vs {reference}",
        "lfc": _metric(f"{prefix}_lfc_pearson"),
        "real_de": _metric(f"{prefix}_real_de_genes", decimals=0),
        "synthetic_de": _metric(f"{prefix}_synthetic_de_genes", decimals=0),
        "overlap": _metric(f"{prefix}_overlap_de_genes", decimals=0),
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
        })

attribute = pd.read_csv(artifact_dir / "attribute_inference.csv")
attribute["excess"] = (
    attribute["training_attack_success_rate"]
    - attribute["control_attack_success_rate"]
)
ranked_secrets = (
    attribute.groupby("secret")["excess"].mean().nlargest(5).index.tolist()
)
fig, ax = plt.subplots(figsize=(5.2, 3.1), dpi=300)
for attack, offset, color, label in (
    ("clinical_qi", -0.19, "#16845a", "Clinical QI"),
    ("standard", 0.19, MODEL_COLORS["dbtwin"], "Clinical QI + genes"),
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
            widths=0.32, patch_artist=True, manage_ticks=False, whis=(0, 100),
            medianprops={"color": "#333333"},
        )
        for box in boxes["boxes"]:
            box.set(facecolor=color, edgecolor="#333333", linewidth=0.8)
        for position, fold_values in available:
            ax.scatter(np.full(len(fold_values), position), fold_values,
                       color="#333333", s=7, zorder=3)
    ax.plot([], [], color=color, linewidth=7, label=label)
ax.set_xticks(range(1, len(ranked_secrets) + 1), [
    ("Disease target" if secret == target else
     secret.removeprefix("Sample Characteristic[").removesuffix("]"))
    for secret in ranked_secrets
], rotation=30, ha="right")
ax.set(xlim=(0.5, len(ranked_secrets) + 0.5), ylim=(0, 0.5),
       ylabel="Training - control success")
ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.grid(axis="y", alpha=0.2)
fig.tight_layout()
fig.savefig(fig_dir / "attribute_inference.png")
plt.close(fig)


report_data = {
    "metrics": {name: _metric(name) for name in metric_names},
    "cohort": {
        "name": args.cohort_name or args.input_dir.name,
        "target": target,
        "real_samples": str(len(expression)),
        "genes": str(len(expression.columns)),
        "splits": str(len(train_sizes)),
        "train_samples": _range(train_sizes),
        "test_samples": _range(test_sizes),
        "synthetic_samples": _range([len(values) for values in synthetic_targets]),
        "classes": ", ".join(map(str, clinical[target].dropna().unique())),
        "n_genes_aux": str(n_genes_aux),
        "n_genes_secret": str(n_genes_secret),
        "contrasts": "; ".join(f"{a} vs {b}" for a, b in contrasts),
        "class_distribution": class_distribution,
    },
    "de_results": de_results,
    "shap_results": shap_results,
    "figure_split": str(sorted(manifest["split"].astype(str).unique())[0]),
    "figures": {
        "coexpression": _figure("coexp_synthetic.png"),
        "umap": _figure("umap.png"),
        "clinical_fidelity": _figure("clinical_fidelity.png"),
        "differential_expression": [
            _template_path(path) for path in sorted(fig_dir.glob("de_*.png"))
        ],
        "shap": [
            _template_path(path)
            for path in sorted(fig_dir.glob("SHAP_Synthetic*.png"))
        ],
        "dcr": _figure("dcr_dbtwin.png"),
        "attribute_inference": _figure("attribute_inference.png"),
    },
}

data_path = artifact_dir / "report_data.json"
data_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")

try:
    report_paths = (template.resolve(), args.output_dir.resolve())
    typst_root = Path(os.path.commonpath(report_paths))
except ValueError as error:
    message = "The report template and output directory must be on the same drive."
    raise SystemExit(message) from error

try:
    subprocess.run(
        (
            "typst",
            "compile",
            "--root",
            typst_root,
            "--input",
            f"data={_template_path(data_path)}",
            template,
            args.output_dir / "benchmark_report.pdf",
        ),
        check=True,
    )
except FileNotFoundError as error:
    raise SystemExit("Typst must be installed and available on PATH.") from error


