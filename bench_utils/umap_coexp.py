"""Compute UMAP and gene coexpression benchmark artifacts."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import rbf_kernel


N_COEXPRESSION_GENES = 500
UMAP_RANDOM_STATE = 42


def _median_absolute_deviation(values: np.ndarray) -> np.ndarray:
    medians = np.median(values, axis=0)
    return np.median(np.abs(values - medians), axis=0)


def _finite_values(frame: pd.DataFrame, columns: np.ndarray) -> np.ndarray:
    return np.nan_to_num(
        frame.loc[:, columns].to_numpy(dtype=float),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def _mmd(real_embedding: np.ndarray, synthetic_embedding: np.ndarray) -> float:
    squared = (
        rbf_kernel(real_embedding).mean()
        + rbf_kernel(synthetic_embedding).mean()
        - 2 * rbf_kernel(real_embedding, synthetic_embedding).mean()
    )
    return float(np.sqrt(np.abs(squared)))


def compute_umap_artifacts(
    df_trn: pd.DataFrame,
    synthetic_frames: dict[str, pd.DataFrame],
    gexp_cols: np.ndarray,
    target: str,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, float]]:
    import umap

    real_values = _finite_values(df_trn, gexp_cols)
    gene_mad = _median_absolute_deviation(real_values)
    selected = np.flatnonzero(gene_mad > np.percentile(gene_mad, 40))
    if selected.size == 0:
        selected = np.argsort(-gene_mad)[: min(2, len(gexp_cols))]
    selected_genes = gexp_cols[selected]

    matrices = {"real": _finite_values(df_trn, selected_genes)}
    matrices.update(
        {
            suffix: _finite_values(frame, selected_genes)
            for suffix, frame in synthetic_frames.items()
        }
    )
    model_order = list(synthetic_frames)
    combined = np.vstack([matrices["real"], *(matrices[key] for key in model_order)])
    embedding = umap.UMAP(
        n_components=3,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        n_jobs=1,
        random_state=UMAP_RANDOM_STATE,
    ).fit_transform(combined)[:, :2]

    offsets = np.cumsum([0, *(len(matrices[key]) for key in ["real", *model_order])])
    real_embedding = embedding[offsets[0] : offsets[1]]
    artifacts: dict[str, dict[str, np.ndarray]] = {}
    mmd_values: dict[str, float] = {}
    for index, suffix in enumerate(model_order, start=1):
        synthetic_embedding = embedding[offsets[index] : offsets[index + 1]]
        artifacts[suffix] = {
            "real": real_embedding.astype(np.float32),
            "synthetic": synthetic_embedding.astype(np.float32),
            "real_target": df_trn[target].to_numpy(),
            "synthetic_target": synthetic_frames[suffix][target].to_numpy(),
        }
        mmd_values[suffix] = _mmd(real_embedding, synthetic_embedding)
    return artifacts, mmd_values


def _correlation_matrix(frame: pd.DataFrame, genes: np.ndarray) -> np.ndarray:
    values = _finite_values(frame, genes)
    # Counts are non-negative; clipping makes log1p well-defined for continuous
    # baselines that can generate negative values.
    values = np.log1p(np.clip(values, 0.0, None))
    matrix = np.corrcoef(values.T)
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)


def compute_coexpression_artifacts(
    df_trn: pd.DataFrame,
    synthetic_frames: dict[str, pd.DataFrame],
    gexp_cols: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    real_values = _finite_values(df_trn, gexp_cols)
    gene_mad = _median_absolute_deviation(real_values)
    top_indices = np.argsort(-gene_mad)[:N_COEXPRESSION_GENES]
    top_genes = gexp_cols[top_indices]

    real_matrix = _correlation_matrix(df_trn, top_genes)
    distance = np.clip(1.0 - real_matrix, 0.0, 2.0)
    order = leaves_list(linkage(squareform(distance, checks=False), method="average"))
    ordered_genes = top_genes[order]
    real_matrix = real_matrix[np.ix_(order, order)]
    upper = np.triu_indices(len(ordered_genes), k=1)

    matrices: dict[str, np.ndarray] = {}
    correlations: dict[str, float] = {}
    for suffix, frame in synthetic_frames.items():
        synthetic_matrix = _correlation_matrix(frame, ordered_genes)
        matrices[suffix] = synthetic_matrix.astype(np.float32)
        correlations[suffix] = float(
            spearmanr(real_matrix[upper], synthetic_matrix[upper]).statistic
        )
    return (
        ordered_genes,
        real_matrix.astype(np.float32),
        matrices,
        correlations,
    )
