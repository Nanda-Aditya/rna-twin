"""Differential expression with PyDESeq2."""

from joblib import parallel_backend
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats


def gexp_de(frame, column, contrast):
    """DESeq2 Wald test of `contrast` = (case, reference) in `column` of samples x genes counts.

    `frame` holds the gene count columns and `column`. Samples with missing `column` are
    dropped, counts are rounded to integers and the design is `~ column`. Cook's-distance
    outliers are replaced and refit as in R DESeq2 (`refit_cooks=True`, for levels with at
    least 7 samples); genes still flagged, and genes removed by independent filtering, have
    padj NaN. Returns one row per gene: `log2FoldChange` (case relative to reference, not
    shrunk) and `padj` (Benjamini-Hochberg adjusted Wald p-value).
    """
    frame = frame.loc[frame[column].notna()]
    labels = frame[column].astype(str)
    case, reference = map(str, contrast)
    if not {case, reference} <= set(labels):
        raise ValueError(f"Contrast {contrast} is absent from {column}")
    # A fixed factor name keeps the design formula valid for any column name.
    dds = DeseqDataSet(
        counts=frame.drop(columns=column).round().astype(int),
        metadata=labels.astype("category").to_frame("condition"),
        design="~ condition", refit_cooks=True)
    with parallel_backend("loky", n_jobs=6):
        dds.deseq2()
    stats = DeseqStats(dds, contrast=["condition", case, reference], quiet=True)
    stats.summary()
    return stats.results_df[["log2FoldChange", "padj"]]
