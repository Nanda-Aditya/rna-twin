"""
Distance-to-closest-record (DCR) privacy metrics for synthetic expression data.

Expression is log2(CPM + 1) normalized (`expression.log_cpm`) and z-scored with training-set
statistics (`_prep_expression`) before Euclidean nearest-neighbour distances (`_knn_euclidean`).
"""


import numpy as np
from sklearn.neighbors import NearestNeighbors


# ──────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────

def _prep_expression(expression, fit_stats=None):
    """z-score a samples x genes frame (missing values as 0). Returns (matrix, fit_stats).

    `fit_stats` = (mean, sd) per gene; when None they are fitted on `expression` itself, with
    constant genes given sd 1.
    """
    X = expression.fillna(0).to_numpy(dtype=float)
    if fit_stats is None:
        mu, sd = X.mean(0), X.std(0)
        sd[sd == 0] = 1.0
        fit_stats = (mu, sd)
    mu, sd = fit_stats
    return (X - mu) / sd, fit_stats

def _knn_euclidean(synth, ref, k=2):
    """Return (n_synth, k) matrix of Euclidean distances to k nearest neighbours."""
    nn = NearestNeighbors(n_neighbors=min(k, len(ref)), metric="euclidean", algorithm="auto")
    nn.fit(ref)
    dists, _ = nn.kneighbors(synth)
    return dists                                                  # (n_synth, k)

def compute_dcr(x_trn, x_syn):
    """Compare synthetic-to-training DCR with the training set's own nearest-neighbour distances.

    `x_trn` and `x_syn` are samples x genes log2(CPM + 1) frames. `d_st` is each synthetic
    sample's distance to its closest training sample; `d_tt` is each training sample's distance
    to its nearest other training sample (returned under the key "d_stst"). `p5_ratio` =
    p5(d_st) / p5(d_tt); values well below 1 suggest memorised records.
    """
    E_trn, estats = _prep_expression(x_trn)
    E_syn, _      = _prep_expression(x_syn, fit_stats=estats)

    d_st= _knn_euclidean(E_syn, E_trn, k=1)
    d_tt= _knn_euclidean(E_trn, E_trn, k=2)[:,1]
    p5_ratio= np.percentile(d_st,5)/ np.percentile(d_tt,5)
    expr_dists = {"d_st": d_st, "d_stst": d_tt,
                "p5_ratio": p5_ratio}
    return expr_dists
