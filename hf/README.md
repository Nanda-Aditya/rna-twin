---
license: other
license_name: dbtwin-dua
license_link: https://huggingface.co/datasets/NandaAditya/rna-twin#license-and-data-use-agreement
pretty_name: 'rna-twin: synthetic bulk RNA-seq cohorts'
language:
- en
task_categories:
- tabular-classification
tags:
- biology
- genomics
- rna-seq
- gene-expression
- transcriptomics
- differential-expression
- synthetic-data
- privacy
- tabular
- healthcare
size_categories:
- n<1K
extra_gated_heading: Accept the dbTwin Data Use Agreement
extra_gated_description: Read the agreement at the end of this card before requesting access.
extra_gated_prompt: 'By requesting access you agree to the Data Use Agreement: noncommercial use only
  and no reverse engineering of the generator.
  Full text: https://huggingface.co/datasets/NandaAditya/rna-twin#license-and-data-use-agreement'
extra_gated_button_content: Agree and access
extra_gated_fields:
  Full name: text
  Affiliation: text
  Organization type:
    type: select
    options:
    - Academic
    - Pharma/Biotech
    - AI company
    - Hospital/Foundation
    - Government
    - Individual/Student
    - Other
  Intended use: text
  I agree to the Data Use Agreement: checkbox
  May dbTwin contact you about future releases? (optional consent):
    type: select
    options:
    - 'No'
    - 'Yes'
configs:
- config_name: pediatric_risk
  data_files:
  - split: full
    path: pediatric_risk/syn_pediatric_risk.csv
  - split: fold1_synthetic
    path: pediatric_risk/artifacts/syn_pediatric_risk_fold_1.csv
  - split: fold2_synthetic
    path: pediatric_risk/artifacts/syn_pediatric_risk_fold_2.csv
  - split: fold3_synthetic
    path: pediatric_risk/artifacts/syn_pediatric_risk_fold_3.csv
  - split: fold4_synthetic
    path: pediatric_risk/artifacts/syn_pediatric_risk_fold_4.csv
  - split: fold5_synthetic
    path: pediatric_risk/artifacts/syn_pediatric_risk_fold_5.csv
- config_name: sepsis
  data_files:
  - split: full
    path: sepsis/syn_sepsis.csv
  - split: fold1_synthetic
    path: sepsis/artifacts/syn_sepsis_fold_1.csv
  - split: fold2_synthetic
    path: sepsis/artifacts/syn_sepsis_fold_2.csv
  - split: fold3_synthetic
    path: sepsis/artifacts/syn_sepsis_fold_3.csv
  - split: fold4_synthetic
    path: sepsis/artifacts/syn_sepsis_fold_4.csv
  - split: fold5_synthetic
    path: sepsis/artifacts/syn_sepsis_fold_5.csv
- config_name: tcga_luad
  data_files:
  - split: full
    path: tcga_luad/syn_tcga_luad.csv
  - split: fold1_synthetic
    path: tcga_luad/artifacts/syn_tcga_luad_fold_1.csv
  - split: fold2_synthetic
    path: tcga_luad/artifacts/syn_tcga_luad_fold_2.csv
  - split: fold3_synthetic
    path: tcga_luad/artifacts/syn_tcga_luad_fold_3.csv
  - split: fold4_synthetic
    path: tcga_luad/artifacts/syn_tcga_luad_fold_4.csv
  - split: fold5_synthetic
    path: tcga_luad/artifacts/syn_tcga_luad_fold_5.csv
---

# rna-twin

Synthetic bulk RNA-seq cohorts with linked clinical variables for three public studies:
pediatric IBD (RISK, GSE57945), sepsis (GSE184900), and TCGA lung adenocarcinoma. We made them
with [dbTwin](https://dbtwin.com). Each cohort has one full synthetic dataset, five
training-fold datasets, and a PDF report comparing the synthetic data with the real data.

![Sepsis: joint real/synthetic UMAP and gene co-expression](assets/qa_preview.png)

*Sepsis. (A) Real and synthetic samples in one UMAP fit. (B) Gene-gene correlations for the
500 most variable genes: real below the diagonal, synthetic above.*

[Benchmark code](https://github.com/Nanda-Aditya/rna-twin) ·
Preprint (forthcoming) <!-- TODO: link the companion preprint DOI once it is assigned. --> ·
[HF collection](https://huggingface.co/collections/NandaAditya/dbtwin-rna-seq-benchmarks-6ae363d8070a32e0b12e71f2) ·
[Data Use Agreement](#license-and-data-use-agreement)

## Get the data

1. Log in, read the [Data Use Agreement](#license-and-data-use-agreement) at the end of this
   page, and click **Agree and access**.
2. Log in from the terminal with the same account: `hf auth login`. If the CLI isn't installed,
   use `uvx --from huggingface-hub hf auth login`.
3. Load a cohort. Each config (`pediatric_risk`, `sepsis`, `tcga_luad`) has a `full` split and
   `fold1_synthetic` through `fold5_synthetic`.

```python
from datasets import load_dataset

full = load_dataset("NandaAditya/rna-twin", "sepsis", split="full",
                    revision="v1.0.0", token=True)
fold1 = load_dataset("NandaAditya/rna-twin", "sepsis", split="fold1_synthetic",
                     revision="v1.0.0", token=True)
```

Or with pandas:

```python
import pandas as pd
from huggingface_hub import hf_hub_download

path = hf_hub_download("NandaAditya/rna-twin", "sepsis/syn_sepsis.csv",
                       repo_type="dataset", revision="v1.0.0")
df = pd.read_csv(path, index_col=0)
```

To rerun the benchmark, clone the [GitHub repo](https://github.com/Nanda-Aditya/rna-twin) and
run `uv run python get_data.py`. It downloads these files into `datasets/<cohort>/dbTwin/`, and
then `uv run python scripts/run_pipe.py sepsis` reruns the sepsis benchmark.

## Files

```text
<cohort>/syn_<cohort>.csv                        full synthetic cohort
<cohort>/artifacts/syn_<cohort>_fold_<1-5>.csv   synthetic training folds
reports/<cohort>.pdf                             benchmark report
assets/qa_preview.png
LICENSE                                          CC BY-NC 4.0
DUA.md                                           Data Use Agreement
```

Each `<cohort>/` folder has the same layout as `datasets/<cohort>/dbTwin/` in the benchmark
repo.

Each row is a synthetic sample. The first, unnamed column is a row index, not a gene or a
patient ID, so read it with `index_col=0`. The other columns are expression values and
clinical variables under their source names: Ensembl gene IDs, plus some gene symbols in
TCGA-LUAD. Clinical values can be missing. RISK and sepsis expression values are read counts.
TCGA-LUAD values are integers on the scale of their source, upper-quartile-normalized RSEM
(each sample's 75th percentile of non-zero genes is about 1,000), so they aren't read counts.

| Cohort | Full rows | Fold rows (1-5) | Genes | Clinical columns | Prediction target | Anchors |
|---|---:|---|---:|---:|---|---|
| `pediatric_risk` | 322 | 257, 257, 258, 258, 258 | 43,790 | 4 | `sample_disease` | disease, age, sex |
| `sepsis` | 152 | 121, 121, 122, 122, 122 | 27,097 | 4 | `Classification` | classification, age, sex |
| `tcga_luad` | 510 | 406, 406, 406, 407, 407 | 19,719 | 10 | `clinical_stage_clean` | EGFR, KRAS, stage, age, sex |

The clinical columns are:

- RISK: `sample_disease`, `Sample Characteristic[sex]`,
  `Sample Characteristic[age at diagnosis years]`, `Sample Characteristic[organism part]`
- Sepsis: `age`, `sex`, `ethnicity`, `Classification`
- TCGA-LUAD: `clinical_os_months`, `clinical_os_status`, `clinical_age`, `clinical_sex`,
  `clinical_egfr_mut`, `clinical_kras_mut`, `clinical_alk_mut`, `clinical_tp53_mut`,
  `clinical_stk11_mut`, `clinical_stage_clean`

Use the full file for clinical, co-expression, UMAP, and DE comparisons. Each fold file was
generated from one real training split only, so compare it with the matching held-out real
split. The split manifests are in the GitHub repo. Don't pool the five folds into one cohort.

## How the data was made

dbTwin is a proprietary, non-deep-learning generator. It takes a real count matrix, sample
metadata, and anchor columns, which are the variables you most want preserved. It generates
synthetic samples with both expression and clinical values. The full cohort was generated
from all real samples. Each fold dataset was generated from one training split, with the
held-out patients left out. The generator isn't included here or in the GitHub repo.

## Benchmark results

From the reports in `reports/` (November 2–3, 2026). Values with ± are mean ± SD over five outer
splits. ✅ marks a pass against the report's fixed thresholds.

| | Pass if | Pediatric RISK | Sepsis | TCGA-LUAD |
|---|---|---:|---:|---:|
| Co-expression Spearman ρ | ≥ 0.8 | 0.996 ✅ | 0.965 ✅ | 0.940 ✅ |
| UMAP MMD | ≤ 0.1 | 0.022 ✅ | 0.037 ✅ | 0.020 ✅ |
| TRTR ROC-AUC (real reference) | | 0.880 ± 0.032 | 0.894 ± 0.032 | 0.573 ± 0.036 |
| TSTR ROC-AUC | ≥ TRTR − 0.05 | 0.878 ± 0.035 ✅ | 0.893 ± 0.027 ✅ | 0.573 ± 0.044 ✅ |
| SHAP Pearson r | ≥ 0.8 | 0.949 ± 0.012 ✅ | 0.881 ± 0.043 ✅ | 0.828 ± 0.056 ✅ |

TRTR trains on real data and TSTR on the matching synthetic fold. Both are scored on the same
held-out real patients.

| Cohort | DE contrast | Real DE genes | log2FC r (≥ 0.8) | Slope | Recall (≥ 80%) |
|---|---|---:|---:|---:|---:|
| Pediatric RISK | Crohn's disease vs control | 5,641 | 0.993 ✅ | 1.00 | 96% ✅ |
| Pediatric RISK | Ulcerative colitis vs control | 1,169 | 0.980 ✅ | 0.95 | 82% ✅ |
| Sepsis | SepsisBSI vs No-Sepsis | 2,743 | 0.994 ✅ | 1.01 | 97% ✅ |
| TCGA-LUAD | KRAS, 0 vs 1 | 1,121 | 0.961 ✅ | 0.94 | 92% ✅ |
| TCGA-LUAD | EGFR, 0 vs 1 | 1,369 | 0.970 ✅ | 0.92 | 93% ✅ |
| TCGA-LUAD | STK11, 0 vs 1 (not an anchor, not graded) | 2,586 | 0.883 | 0.70 | 69% |

Real DE genes have padj < 0.05 and |log2FC| > 0.5. Recall is the share of them that are also
significant in the synthetic data (padj < 0.075, |log2FC| > 0.4). STK11 wasn't an anchor, and
it shows: recall drops to 69% and effects shrink by about 30%.

| | Pediatric RISK | Sepsis | TCGA-LUAD |
|---|---:|---:|---:|
| DCR 5th-percentile ratio | 0.916 ± 0.011 | 1.056 ± 0.035 | 1.047 ± 0.012 |
| Attribute inference excess, clinical QIs | 0.029 | 0.066 | 0.028 |
| Attribute inference excess, QIs + 100 genes | 0.094 | 0.112 | 0.143 |
| Secrets with excess > 0.2 (clinical / + genes) | 0 / 1 | 0 / 0 | 0 / 3 |

A DCR ratio near 1 means synthetic records are no closer to training patients than training
patients are to each other. Excess success is an Anonymeter attack's success on training
patients minus its success on held-out patients, summarized as the median of the 10 highest
per-secret means.

## Uses and limitations

The data is meant for noncommercial method development, benchmarking, teaching, pipeline
prototyping, and hypothesis generation. Confirm anything that matters on real data. It isn't
validated for diagnosis, treatment decisions, or clinical or regulatory evidence.

The results above cover only the tested targets, contrasts, gene subsets, and classifier. We
haven't tested survival models, pathway enrichment, or external cohorts. The cohorts are
small, subgroup estimates can be unstable, and biases in the source studies carry over.

The records are synthetic and were generated from public, de-identified data. We measured
disclosure risk with DCR and two attribute-inference attacks. These are empirical tests, not
differential privacy.

The real data comes from GEO ([GSE57945](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE57945),
[GSE184900](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE184900)) and TCGA
([cBioPortal PanCancer Atlas](https://www.cbioportal.org/study/summary?id=luad_tcga_pan_can_atlas_2018)).
Its original terms apply, so please cite the original studies. The TCGA results shown here are in whole or part based upon
data generated by the TCGA Research Network: https://www.cancer.gov/tcga.

## Citation

If you share the data or anything made from it, include this attribution and say what you
changed:

> rna-twin v1.0.0 © 2026 dbTwin, Inc., CC BY-NC 4.0, https://huggingface.co/datasets/NandaAditya/rna-twin

Please cite the benchmark repository and the dataset:

```bibtex
@software{nanda2026rnatwin,
  title  = {rna-twin: synthetic bulk RNA-seq cohorts and benchmark},
  author = {Nanda, Aditya and Saha, Somdutta},
  year   = {2026},
  url    = {https://github.com/Nanda-Aditya/rna-twin}
}

@misc{rnatwin2026,
  title     = {rna-twin: synthetic bulk RNA-seq cohorts},
  author    = {Nanda, Aditya and Saha, Somdutta},
  publisher = {Hugging Face},
  year      = {2026},
  version   = {v1.0.0},
  url       = {https://huggingface.co/datasets/NandaAditya/rna-twin}
}
```

## Changelog

- **v1.0.0, November 3–4, 2026:** first release. The CSVs are the files the benchmark reports
  in `reports/` were computed from.

The examples above pin `revision="v1.0.0"`.

## License and Data Use Agreement

Version 1.0, October 30, 2026

This agreement is between you and dbTwin, Inc. It covers the synthetic CSVs in the Hugging Face
dataset `NandaAditya/rna-twin`, both the full cohorts and the training-fold datasets. You accept
it by clicking **Agree and access** on the dataset page.

The synthetic data is © 2026 dbTwin, Inc. and licensed under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). The full text is in the
`LICENSE` file. The terms below are in addition to that license. They don't change it, and
they don't take away any rights it gives you.

1. Use the data only for noncommercial purposes, as CC BY-NC 4.0 defines them, unless dbTwin
   has given you a commercial license.
2. Don't try to reconstruct dbTwin's generation method, or use the data to train a model meant
   to imitate it, without our written permission. Training other models, benchmarking,
   comparing generators, running privacy tests, and publishing your results, good or bad, are
   all fine.
3. The data is provided as is, without warranties of any kind. It isn't validated for
   diagnosis, treatment, or other clinical decisions.
4. We may revoke your Hugging Face access if you break these terms.
5. If we update this agreement, the new version applies only to access granted after the
   update.

Hugging Face shares your account details and form answers with dbTwin. We use them to manage
access and to understand how the data is used, and we keep them only as long as we need them
for that. We'll email you about new releases only if you chose Yes on the form. To see,
correct, or delete your information, or to change that choice, email aditya@dbtwin.ai.

Questions and commercial licensing: aditya@dbtwin.ai.
