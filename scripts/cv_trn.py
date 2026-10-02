#!/usr/bin/env python3
"""Select elastic-net C values using only each split's training data."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("input_dir", type=Path)
parser.add_argument("output_dir", type=Path)
parser.add_argument("--target", required=True)
parser.add_argument("--force", action="store_true")
args = parser.parse_args()

cache = args.output_dir / "cv_results.csv"
if cache.exists() and not args.force:
    print(f"Reusing {cache}")
    raise SystemExit(0)

C_VALUES = np.logspace(-4, 1, 51)
L1_RATIO = 0.05
SPARSITY_BAND = (0.10, 0.40)

clinical = pd.read_csv(args.input_dir / "clinical.csv", index_col=0)
expression = pd.read_csv(args.input_dir / "expression.csv", index_col=0)
manifest = pd.read_csv(args.input_dir / "split_manifest.csv", dtype={"sample_id": str})
clinical.index = clinical.index.map(str)
expression.index = expression.index.map(str)
real = expression.join(clinical, how="inner")
genes = expression.columns.to_numpy()
results = []

for split, rows in manifest.groupby("split", sort=True):
    sample_ids = rows.loc[
        rows["partition"].str.lower() == "train", "sample_id"
    ]
    trn = real.loc[sample_ids]
    n_features = max(1, min(270, int(len(trn) * 0.9), len(genes)))
    n_splits = min(5, trn[args.target].value_counts().min())
    if n_splits < 2:
        raise ValueError(f"Split {split} has fewer than two samples in a target class")

    folds = []
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_index, valid_index in cv.split(trn, trn[args.target]):
        inner_trn = trn.iloc[train_index]
        inner_valid = trn.iloc[valid_index]
        mad = inner_trn[genes].sub(inner_trn[genes].median()).abs().median()
        selected_genes = mad.nlargest(n_features).index
        scaler = StandardScaler()
        x_train = scaler.fit_transform(np.log1p(inner_trn[selected_genes]))
        x_valid = scaler.transform(np.log1p(inner_valid[selected_genes]))
        folds.append((x_train, inner_trn[args.target], x_valid,
                      inner_valid[args.target]))

    def fit_candidate(C):
        aucs, active_fractions = [], []
        for x_train, y_train, x_valid, y_valid in folds:
            clf = LogisticRegression(
                penalty="elasticnet", solver="saga", C=C,
                l1_ratio=L1_RATIO, class_weight="balanced", max_iter=6000,
                random_state=42,
            ).fit(x_train, y_train)
            probabilities = clf.predict_proba(x_valid)
            auc = (
                roc_auc_score(y_valid, probabilities[:, 1])
                if len(clf.classes_) == 2 else
                roc_auc_score(
                    y_valid, probabilities, labels=clf.classes_,
                    multi_class="ovr", average="macro",
                )
            )
            active = np.any(np.abs(np.atleast_2d(clf.coef_)) > 1e-6, axis=0)
            aucs.append(auc)
            active_fractions.append(active.mean())
        return {
            "split": split,
            "C": C,
            "l1_ratio": L1_RATIO,
            "mean_auc": np.mean(aucs),
            "std_auc": np.std(aucs),
            "n_folds": len(aucs),
            "mean_active_fraction": np.mean(active_fractions),
            "std_active_fraction": np.std(active_fractions),
            "folds_in_sparsity_band": all(
                SPARSITY_BAND[0] <= fraction <= SPARSITY_BAND[1]
                for fraction in active_fractions
            ),
            "selected": False,
        }

    split_results = Parallel(n_jobs=-1)(
        delayed(fit_candidate)(C) for C in C_VALUES
    )
    results.extend(split_results)

eligible = {}
for C in C_VALUES:
    candidates = [result for result in results if result["C"] == C]
    if all(result["folds_in_sparsity_band"] for result in candidates):
        eligible[C] = np.average(
            [result["mean_auc"] for result in candidates],
            weights=[result["n_folds"] for result in candidates],
        )
if not eligible:
    raise ValueError(
        "No C has every fold's active fraction in "
        f"{SPARSITY_BAND[0]:.0%}-{SPARSITY_BAND[1]:.0%}"
    )
best_C = max(eligible, key=lambda C: (eligible[C], -C))
for result in results:
    result["selected"] = result["C"] == best_C
print(f"C={best_C:.6g}, l1_ratio={L1_RATIO}, mean AUC={eligible[best_C]:.3f}")

args.output_dir.mkdir(parents=True, exist_ok=True)
pd.DataFrame(results).to_csv(cache, index=False)
