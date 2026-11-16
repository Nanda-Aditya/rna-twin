# Sepsis (GSE189400)

Whole-blood RNA-seq from critically ill adults in the EARLI cohort: 152 subjects, 27,097
genes, 4 clinical columns. `real_data/` holds the real cohort and its split manifest; `dbTwin/`
holds the benchmark settings, report, and saved outputs. The synthetic CSVs are on
[Hugging Face](https://huggingface.co/datasets/NandaAditya/rna-twin).

## Source

GEO [GSE189400](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE189400), the
whole-blood (PAXgene) RNA-seq subseries of SuperSeries
[GSE189403](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE189403)
(Kalantar et al., Nat Microbiol 2022). The series has 221 subjects: 92 `No-Sepsis`, 60
`SepsisBSI`, and 69 `Sepsisperiph` (sepsis from a non-bloodstream source). The 69
`Sepsisperiph` subjects are excluded, and all 27,097 genes are kept.

## Expression (`real_data/expression.csv.gz`)

- Integer read counts, with versioned Ensembl gene IDs (e.g. `ENSG00000000003.14`).
- Rows are GEO sample accessions (`GSM...`), columns are genes.

## Clinical columns (`real_data/metadata.csv`)

| Column | Values |
|---|---|
| `Classification` | `SepsisBSI` (60; sepsis with bloodstream infection), `No-Sepsis` (92) |
| `age` | years |
| `sex` | `Female` / `Male` |
| `ethnicity` | `LatinX` / `Not_LatinX`; missing for one subject |

Only the `SepsisBSI` and `No-Sepsis` groups are included.

## Terms and citation

GEO data are public. Please cite Kalantar et al., "Integrated host-microbe plasma metagenomics
for sepsis diagnosis in a prospective cohort of critically ill adults", Nat Microbiol 2022.
