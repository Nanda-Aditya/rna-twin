"""Elastic-net logistic regression trained on real (TRTR) or synthetic (TSTR) data.

Both models are scored on the same held-out real samples, with per-class metrics,
coefficients and linear SHAP values.
"""

import numpy as np
import shap
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, label_binarize

from bench_utils import config
from bench_utils.expression import top_variable_genes


def make_model(C, l1_ratio):
    """Median imputation, standardization and balanced elastic-net logistic regression (saga)."""
    return make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(solver="saga", C=C, l1_ratio=l1_ratio, class_weight="balanced",
                           max_iter=5 if config.demo_run else 6000, random_state=42))


def fit_classifier(x_trn, y_trn, x_tst, y_tst, C, l1_ratio):
    """Fit `make_model` on (`x_trn`, `y_trn`) and evaluate it on (`x_tst`, `y_tst`).

    Returns a dict with:
        - classes: sorted class labels, the order of every per-class output
        - auc / prauc: per-class one-vs-rest ROC-AUC and average precision (PR-AUC)
        - pcr: per-class recall
        - coeff: one row per class (one row for a binary target): intercept, then coefficients
        - shap: linear SHAP values (log-odds) of the standardized test features against the
          training background; (n_test, n_features) for the positive class `classes[1]` of a
          binary target, (n_test, n_features, n_classes) otherwise
    """
    model = make_model(C, l1_ratio).fit(x_trn, y_trn)
    clf, classes = model[-1], model[-1].classes_
    proba = model.predict_proba(x_tst)
    onehot = label_binarize(y_tst, classes=classes)
    if len(classes) == 2:
        onehot = np.hstack([1 - onehot, onehot])
    background, test = (model[:-1].transform(x) for x in (x_trn, x_tst))
    masker = shap.maskers.Independent(background, max_samples=len(background))
    return {
        "classes": classes,
        "auc": roc_auc_score(onehot, proba, average=None),
        "prauc": average_precision_score(onehot, proba, average=None),
        "pcr": recall_score(y_tst, classes[proba.argmax(axis=1)], labels=classes, average=None),
        "coeff": np.hstack([np.atleast_2d(clf.intercept_).T, np.atleast_2d(clf.coef_)]),
        "shap": shap.LinearExplainer(clf, masker).shap_values(test),
    }


def train_trtr_tstr(x_trn, y_trn, x_syn, y_syn, x_tst, y_tst, n_features, C, l1_ratio):
    """Train on real training (TRTR) and synthetic (TSTR) data; score both on real test data.

    `x_*` are samples x genes log2(CPM + 1) frames. Features are the `n_features` genes with
    the highest real-training variance; C and l1_ratio are fixed (selected by cv_trn.py). Returns the
    real- and synthetic-trained `fit_classifier` results and the selected genes.
    """
    genes = top_variable_genes(x_trn, n_features)
    real, synthetic = (fit_classifier(x[genes], y, x_tst[genes], y_tst, C, l1_ratio)
                       for x, y in ((x_trn, y_trn), (x_syn, y_syn)))
    return real, synthetic, genes.to_numpy()
