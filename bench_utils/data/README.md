# HGNC Ensembl-to-symbol mapping

`hgnc_ensembl_symbols_2026-07-07.tsv.gz` contains the `ensembl_gene_id` and
approved `symbol` columns from the HGNC quarterly snapshot dated 2026-07-07:

https://storage.googleapis.com/public-download-files/hgnc/archive/archive/quarterly/tsv/hgnc_complete_set_2026-07-07.tsv

Rows without an Ensembl gene ID are excluded. Three Ensembl IDs associated with
multiple approved symbols are also excluded so callers retain the original ID
rather than receiving an ambiguous symbol.

HGNC data is released under the Creative Commons Zero (CC0) license:

https://hgnc.genenames.org/about/license/
