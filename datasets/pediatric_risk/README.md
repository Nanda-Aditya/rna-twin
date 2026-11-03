# Pediatric RISK (GSE57945)

Ileal biopsies from treatment-naive children in the RISK pediatric inflammatory bowel disease
inception cohort: 322 samples, 43,790 genes, 4 clinical columns. `real_data/` holds the real
cohort and its split manifest; `dbTwin/` holds the benchmark settings, report, and saved
outputs. The synthetic CSVs are on [Hugging Face](https://huggingface.co/datasets/NandaAditya/rna-twin).

## Source

GEO [GSE57945](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE57945), as processed by
EMBL-EBI Expression Atlas experiment [E-GEOD-57945](https://www.ebi.ac.uk/gxa/experiments/E-GEOD-57945):
`E-GEOD-57945-raw-counts.tsv` and `E-GEOD-57945-experiment-design.tsv`.

## Expression (`real_data/expression.csv.gz`)

- Integer read counts from Expression Atlas, with Ensembl gene IDs without version suffixes.
- Rows are sequencing runs (`SRR...`), columns are genes.
- All 322 runs in the experiment design were kept.
- Genes with a total count of 10 or less across samples were removed.

## Clinical columns (`real_data/metadata.csv`)

| Column | From | Values |
|---|---|---|
| `sample_disease` | `Sample Characteristic[disease]` | `Crohn's disease` (218), `ulcerative colitis` (62), `cntrl` (42; non-IBD controls) |
| `Sample Characteristic[sex]` | same | `female` / `male` |
| `Sample Characteristic[age at diagnosis years]` | `Sample Characteristic[age at diagnosis]` | years, as a number |
| `Sample Characteristic[organism part]` | same | `ileum` for every sample |

## Terms and citation

GEO and Expression Atlas data are public. Please cite Haberman et al., J Clin Invest 2014
(the RNA-seq study), Kugathasan et al., Lancet 2017 (the RISK cohort), and Expression Atlas.
