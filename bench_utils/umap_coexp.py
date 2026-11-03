"""Compute UMAP and gene coexpression benchmark artifacts from log2(CPM + 1) expression."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import rbf_kernel

from bench_utils.expression import top_variable_genes


N_COEXPRESSION_GENES = 500
N_PCS = 100
UMAP_RANDOM_STATE = 42


def _mmd(real_embedding: np.ndarray, synthetic_embedding: np.ndarray) -> float:
    """Biased (V-statistic) RBF-kernel MMD, sqrt(|MMD^2|), with gamma = 1 / n_features."""
    squared = (
        rbf_kernel(real_embedding).mean()
        + rbf_kernel(synthetic_embedding).mean()
        - 2 * rbf_kernel(real_embedding, synthetic_embedding).mean()
    )
    return float(np.sqrt(np.abs(squared)))


def compute_umap_artifacts(
    x_real: pd.DataFrame, x_syn: pd.DataFrame, n_components: int = 2,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Embed real and synthetic expression in one joint UMAP and score their overlap.

    `x_real` and `x_syn` are samples x genes log2(CPM + 1) frames (the full cohorts in
    compute_metrics.py); missing values count as 0. Genes with real variance above its 40th
    percentile are reduced, jointly and without scaling, to `N_PCS` principal components, which
    one UMAP fit (15 neighbors, minimum distance 0.1, Euclidean distance) embeds in
    `n_components` dimensions. Returns the real and synthetic embeddings and their `_mmd`.
    """
    import umap

    variance = x_real.var()
    genes = variance.index[variance > variance.quantile(0.4)]
    combined = pd.concat([x_real[genes], x_syn[genes]]).fillna(0).to_numpy(dtype=float)
    pcs = PCA(min(N_PCS, *combined.shape), random_state=UMAP_RANDOM_STATE).fit_transform(combined)
    embedding = umap.UMAP(
        n_components=n_components,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        n_jobs=1,
        random_state=UMAP_RANDOM_STATE,
    ).fit_transform(pcs)
    real, synthetic = embedding[:len(x_real)], embedding[len(x_real):]
    return real.astype(np.float32), synthetic.astype(np.float32), _mmd(real, synthetic)


def _correlation_matrix(expression: pd.DataFrame) -> np.ndarray:
    """Gene-gene Pearson correlations (missing values as 0); undefined entries become 0."""
    matrix = np.corrcoef(expression.fillna(0).to_numpy(dtype=float).T)
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)


def compute_coexpression_artifacts(
    x_real: pd.DataFrame, x_syn: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compare gene-gene co-expression structure of real and synthetic data.

    Takes the top `N_COEXPRESSION_GENES` genes by real variance of the log2(CPM + 1) frames
    `x_real` and `x_syn`, builds Pearson correlation matrices, and orders genes
    (`ordered_genes`) by average-linkage clustering of the real matrix. Returns the ordered
    genes, the real and synthetic matrices in that order, and the Spearman rho between their
    upper triangles.
    """
    top_genes = top_variable_genes(x_real, N_COEXPRESSION_GENES)
    real_matrix = _correlation_matrix(x_real[top_genes])
    distance = np.clip(1.0 - real_matrix, 0.0, 2.0)
    order = leaves_list(linkage(squareform(distance, checks=False), method="average"))
    ordered_genes = top_genes[order]
    real_matrix = real_matrix[np.ix_(order, order)]
    synthetic_matrix = _correlation_matrix(x_syn[ordered_genes])
    upper = np.triu_indices(len(ordered_genes), k=1)
    rho = float(spearmanr(real_matrix[upper], synthetic_matrix[upper]).statistic)
    return (
        ordered_genes.to_numpy(),
        real_matrix.astype(np.float32),
        synthetic_matrix.astype(np.float32),
        rho,
    )
