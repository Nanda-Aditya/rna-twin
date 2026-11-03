"""Library-size normalization and variable-gene ranking shared by the benchmark metrics."""

import numpy as np


def log_cpm(counts):
    """log2(CPM + 1) of a samples x genes count frame; CPM uses each sample's total over `counts`.

    Negative counts are clipped to 0. Missing counts stay missing and add nothing to the total.
    """
    counts = counts.clip(lower=0)
    library = counts.sum(axis=1)
    return np.log2(counts.div(library.where(library > 0, 1), axis=0) * 1e6 + 1)


def top_variable_genes(expression, n):
    """Columns of `expression` with the `n` largest variances across samples, largest first.

    Applied to log2(CPM + 1), as in the DESeq2 and limma conventions of ranking genes by
    variance of log-normalized expression.
    """
    return expression.var().nlargest(n).index


def n_ml_features(n_samples, n_genes):
    """ML feature count: 90% of training samples, capped at the available genes, at least 1."""
    return max(1, min(int(n_samples * 0.9), n_genes))
