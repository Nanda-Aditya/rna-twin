#import "styles.typ": report-style, report-table

#show: report-style
#set document(title: "Synthetic Clinico-Transcriptomic Data Benchmark Report", author: "dbTwin Inc.")
#show heading.where(level: 1): set text(size: 17pt, weight: "bold")
#show heading.where(level: 2): set text(size: 13pt, weight: "bold")

#let data = json(sys.inputs.at("data"))
#let metric(name) = data.metrics.at(name)
#let cohort = data.cohort
#let figures = data.figures
#let ml-target = cohort.at("ml_target", default: cohort.at("target", default: ""))
#let de-contrasts = data.at("de_contrasts", default: data.de_results)
#let de-results = data.de_results
#let anchor-cols = data.at("anchor_cols", default: cohort.anchor_cols.split(",").map(c => c.trim()))
#let non-anchor-label() = box(
  stroke: 0.5pt + rgb("#e67e22"), inset: 2pt, radius: 2pt,
  text(size: 7pt, fill: rgb("#b85c00"))[non-anchor feature],
)
#let feature-label(name, column) = [
  #name#if not (column in anchor-cols) [\ #non-anchor-label()]
]
#let plot-panel(path, ..options) = stack(
  spacing: 3pt, image(path, ..options),
  ..if not (data.at("figure_columns", default: (:)).at(path, default: ml-target) in anchor-cols) {
    (align(center, non-anchor-label()),)
  } else { () },
)
// Two panels per row; an odd final panel is centred at the same width.
#let panel-grid(panels) = grid(
  columns: (1fr, 1fr), gutter: 12pt,
  ..panels.enumerate().map(((i, panel)) => if calc.odd(panels.len()) and i == panels.len() - 1 {
    grid.cell(colspan: 2, align(center, block(width: 50% - 6pt, panel)))
  } else { panel }),
)
#let de-name(item) = if item.at("categories", default: none) != none {
  str(item.categories.at(0)) + " vs " + str(item.categories.at(1)) + " (" + item.column + ")"
} else {
  item.contrast
}
#let shap-results = data.shap_results
#let table-caption(number, caption) = {
  text(size: 8.5pt)[*Table #number.* #caption]
  v(4pt)
}
#let status-cell(status) = if status == "Pass" {
  text(size: 8pt, weight: "bold", fill: rgb("#1a7f37"), status)
} else if status == "Fail" {
  text(size: 8pt, weight: "bold", fill: rgb("#cf222e"), status)
} else {
  text(size: 8pt)[--]
}
#let summary-entry(name, item, threshold) = (
  text(weight: "medium", name), text(size: 8pt, weight: "bold", item.estimate),
  text(size: 8pt, threshold), status-cell(item.status),
)
#let summary-table(groups) = {
  set text(size: 8.5pt)
  set par(justify: false)
  table(
    columns: (auto,) * 4, align: (left + horizon, ..(center + horizon,) * 3),
    inset: 3.2pt, stroke: (x: 0.5pt + black, y: none), fill: white,
    table.hline(stroke: 0.5pt + black),
    table.header(
      [*Metric*], [*Estimate*], [*Threshold*], [*Pass/Fail*],
      table.hline(stroke: 0.5pt + black),
    ),
    ..groups.map(((title, rows)) => (
      table.cell(colspan: 4, align: center, fill: luma(240))[*#title*],
      ..rows.map(row => summary-entry(..row)).flatten(),
    )).flatten(),
    table.hline(stroke: 0.5pt + black),
  )
}
#let summary-groups = (
  ("Statistical metrics", (
    ("Gene co-expression fidelity (Sp. rho)", metric("coexpression_spearman"), [≥ 0.8]),
    ("UMAP embedding fidelity (MMD)", metric("umap_mmd"), [≤ 0.1]),
  )),
  ("DE and ML metrics", (
    ..(if ml-target in anchor-cols { (
      ("TRTR AUC (real-data reference)", metric("roc_auc_trtr"), [--]),
      ("TSTR AUC", metric("roc_auc_tstr"), [≥ TRTR − 0.05]),
    ) } else { () }),
    ..range(de-results.len()).filter(i => de-contrasts.at(i).column in anchor-cols).map(i => (
      ("DE log2FC Pearson r: " + de-name(de-contrasts.at(i)), de-results.at(i).lfc, [≥ 0.8]),
      ("DE gene recall (relaxed): " + de-name(de-contrasts.at(i)), de-results.at(i).recovery,
        [≥ 80%]),
    )).join(),
    ..(if ml-target in anchor-cols {
      (("SHAP Pearson r", metric("shap_pearson"), [≥ 0.8]),)
    } else { () }),
  )),
)
#let graded = summary-groups.map(((title, rows)) => rows).join().filter(
  row => row.at(1).status in ("Pass", "Fail"))
#let failed = graded.filter(row => row.at(1).status == "Fail")
#let ungraded-de = range(de-results.len()).filter(i => not (de-contrasts.at(i).column in anchor-cols))

#page(footer: none, align(center + horizon)[
  #set par(justify: false)
  #set text(hyphenate: false)
  #text(size: 24pt, weight: "bold")[Synthetic Clinico-Transcriptomic Data Benchmark Report]
  #v(12pt)
  #text(size: 14pt)[Using differential expression, machine learning, statistical fidelity and privacy analyses to benchmark synthetic RNA-seq cohorts]
  #v(36pt)
  #text(size: 12pt)[*Author:* dbTwin, Inc.]
  #v(4pt)
  #text(size: 12pt)[*Email:* #link("mailto:aditya@dbtwin.ai")[aditya\@dbtwin.ai]]
  #v(36pt)
  #text(size: 9pt)[Report generated #data.provenance.date with #data.provenance.pipeline \ #data.provenance.packages]
])

= Executive summary

This report evaluates the quality of the synthetic cohort created from the source cohort (#(cohort.name)) using dbTwin [1], a proprietary synthetic data engine that uses advanced ML methods to build synthetic multimodal transcriptomic data (e.g., RNA-seq counts with linked clinical metadata). This report compares the synthetic cohort against the real cohort across translational analyses: differential expression, predictive ML, gene expression statistics and privacy analyses.

#block(width: 100%, inset: 8pt, radius: 2pt, stroke: 0.5pt + black, fill: luma(245))[
  *Bottom line.* #graded.filter(row => row.at(1).status == "Pass").len() of #graded.len()
  graded fidelity and utility measures passed their thresholds (Table 2)#if failed.len() > 0 [; failed: #failed.map(row => row.at(0)).join("; ")].
  #if ungraded-de.len() > 0 [Not graded (non-anchor columns): #ungraded-de.map(i => [#de-name(de-contrasts.at(i)), log2FC r = #de-results.at(i).lfc.estimate, slope = #de-results.at(i).slope.estimate, relaxed recall #de-results.at(i).recovery.estimate]).join("; ") (Table 3).]
  Privacy is not graded: the DCR 5th-percentile ratio was #metric("privacy_p5_ratio").estimate
  (near or above 1 = no excess proximity to training records); attribute-inference results are in Table 5.
]

dbTwin needs two required inputs (expression.csv and metadata.csv) and, optionally, a list of anchor columns (#raw("anchor_cols")), a user-specified set of translationally meaningful columns that the scientist is interested in. dbTwin supports 7–10 anchor columns for most datasets. Almost all ML and DE analyses on anchor columns produce high-fidelity concordance with real data; columns not explicitly declared as anchor columns produce moderate fidelity.

#v(4pt)
#table-caption(1, [Source cohort (#(cohort.name)) used to generate the synthetic data, and the evaluation design. Files: Table 7.])
#report-table(
  ("Item", "Value"),
  (
    ("Real samples", cohort.real_samples),
    ("Genes", cohort.genes),
    ("Clinical variables", cohort.clinical_vars),
    ("Anchor columns", raw(cohort.anchor_cols)),
    ("ML target (classes)", [#raw(ml-target) (#(cohort.classes))]),
    ("DE contrasts", de-contrasts.map(de-name).join("; ")),
    ("DE genes", [#(cohort.de_genes_tested) tested; #(cohort.de_genes_reported) in the top 80% by mean expression, used in DE tables and figures]),
    ("Outer folds", [#(cohort.splits) (#(cohort.train_samples) real training, #(cohort.test_samples) held-out real samples each)]),
    ("Synthetic samples per fold", cohort.synthetic_samples),
  ),
  column-widths: (auto, auto), width: auto, inset: 3.5pt,
)
#v(4pt)

Global fidelity is assessed using maximum mean discrepancy (MMD) between real and synthetic UMAP embeddings, and by gene--gene co-expression correlation. Discovery fidelity is assessed by differential expression (DE) concordance and the train-on-synthetic, test-on-real (TSTR) paradigm, which tests whether synthetic-trained ML models generalize to held-out real subjects. SHAP concordance then compares gene-level SHAP attributions between real- and synthetic-trained models. Privacy is evaluated separately using a closest-record memorization test and
two attribute-inference attacks (see Results, Table 5).

#v(6pt)
#block(breakable: false)[
#table-caption(2, [Summary of benchmark measures with pass/fail against fixed thresholds (see FAQs). DE, UMAP, and co-expression compare the full release with the full real cohort it was generated from (in-sample). Other values are mean ± SD across outer splits on held-out real samples; SHAP uses valid splits. TRTR is the real-data reference and is not graded. DE gene recall uses the relaxed synthetic threshold (padj < 0.075, |log2FC| > 0.4). Non-anchor columns are omitted (see Bottom line). Privacy results are reported in Table 5.])
#summary-table(summary-groups)
]

#v(10pt)
= Results

Clinical fidelity, global expression structure and DE compare the full synthetic release
with the full real cohort. These comparisons are in-sample: the release was generated from
all real samples, which it is then compared against. Only TSTR, SHAP and the privacy tests
use real samples held out from generation.

== Clinical fidelity

#for panel in figures.clinical_fidelity {
  figure(
    image(panel.path, width: 75%),
    caption: [*Clinical fidelity of the synthetic cohort.* Panels compare full real and dbTwin synthetic distributions of #raw(panel.variable), conditioned on ML-target class.],
  )
}

== Global expression structure

We tested whether dbTwin preserved global statistical and embedding fidelity. 
A joint UMAP embedding provides a qualitative view of whether real and synthetic samples
occupy comparable low-dimensional regions, while MMD summarizes separation in
that projected space. Co-expression instead tests whether pairwise correlation
structure among the most variable genes is retained. 

Full-cohort UMAP MMD was #strong(metric("umap_mmd").estimate). The
co-expression matrices had Spearman correlation
#strong(metric("coexpression_spearman").estimate). Thus, the synthetic cohort retained
the evaluated global geometry and correlation ordering to the degree quantified
by these statistics. The UMAP and heat map below use the full real and synthetic cohorts.
Standard per-gene RNA-seq checks (mean, variance and zero fraction of every gene, and
library size per sample) are in Appendix A (Figure A1).

#if figures.expression_fidelity != none {
  figure(
    image(figures.expression_fidelity, width: 80%),
    caption: [*Global expression structure of the full cohorts.* (A) Joint UMAP of real (dark grey) and dbTwin synthetic (green) samples, using log2(CPM + 1) (counts per million from each sample's total counts) of genes above the 40th percentile of real-cohort variance, reduced to 100 principal components. Interleaved clouds mean synthetic samples occupy the same regions as real ones; the title gives the RBF-kernel MMD between them (0 = indistinguishable; at most 0.1 passes). (B) Pearson correlations of log2(CPM + 1) for the 500 most variable real-cohort genes: real below the diagonal, synthetic above, genes in the same clustered order, so preserved modules appear as mirror-image blocks. The title gives the Spearman correlation of the off-diagonal entries (at least 0.8 passes).],
  )
}

== Differential-expression fidelity

Differential-expression fidelity asks whether a researcher would recover
comparable gene-level effects from the synthetic cohort. For each prespecified
contrast, DESeq2 was applied independently to the full real and synthetic
cohorts. Real DE genes have adjusted p-value below 0.05 and |log2FC| above 0.5.
Agreement was evaluated using Pearson correlation of log2 fold-change among
real DE genes. Synthetic DE genes use two thresholds:

- *Strict:* padj < 0.05 and |log2FC| > 0.5 (same as real).
- *Relaxed:* padj < 0.075 and |log2FC| > 0.4.

#v(8pt)
#block(breakable: false)[
#table-caption(3, [Differential-expression agreement and DE gene counts in the full cohorts. log2FC is the first-named group relative to the second, so a positive value means higher expression in the first-named group (e.g. higher in 0 for "0 vs 1"). Real DE genes have adjusted p-value below 0.05 and |log2FC| above 0.5. log2FC r and slope are the Pearson correlation and least-squares slope of synthetic on real log2FC over all real DE genes; a slope below 1 means synthetic effects are shrunk toward zero. Synthetic DE genes use the strict or relaxed thresholds below. Recall counts real DE genes that are also synthetic DE genes, with the percentage of real DE genes recalled in brackets. Total counts all synthetic DE genes; Precision is the recalled count divided by Total. Per-gene results are in the CSV files listed in Table 7.])
#{
  set text(size: 8.5pt)
  set par(justify: false)
  let rule = 0.5pt + black
  table(
    columns: (auto,) * 7,
    align: center + horizon, inset: 6pt,
    stroke: (x: rule, y: none), fill: white,
    table.hline(stroke: rule),
    table.header(
      table.cell(rowspan: 2)[*Contrast*], table.cell(rowspan: 2)[*log2FC r*],
      table.cell(rowspan: 2)[*log2FC slope*],
      table.cell(rowspan: 2)[*Real DE genes*],table.cell(colspan: 3)[*Synthetic DE genes*],
      table.hline(start: 4, stroke: rule),
      [], [*Strict* \ padj < 0.05 \ |log2FC| > 0.5],
      [*Relaxed* \ padj < 0.075 \ |log2FC| > 0.4],
      table.hline(stroke: rule),
    ),
    ..de-results.enumerate().map(((i, item)) => (
      table.cell(rowspan: 3)[#feature-label(de-name(de-contrasts.at(i)), de-contrasts.at(i).column) \ #text(size: 7pt)[positive = higher in #item.case]],
      table.cell(rowspan: 3)[*#item.lfc.estimate*],
      table.cell(rowspan: 3)[*#item.slope.estimate*],
      table.cell(rowspan: 3)[#item.real_de.estimate],
      [Recall], ..item.synthetic_thresholds.map(t => [*#t.recovered (#t.recovery_percent)*]),
      [Total], ..item.synthetic_thresholds.map(t => [#t.total]),
      [Precision], ..item.synthetic_thresholds.map(t => [*#t.precision*]),
    )).flatten(),
    table.hline(stroke: rule),
  )
}
]

#v(10pt)

The reported correlations and slopes use all real DE genes, regardless of synthetic
significance, so they test whether real effect sizes are reproduced: r measures
agreement in ordering, while the slope shows whether effects keep their real size.
When counting synthetic DE genes, the relaxed threshold (padj < 0.075 and
|log2FC| > 0.4) is recommended to account for weaker significance and reduced statistical power,
particularly in minority classes. The panels below show all real DE genes in
the full cohorts; each subtitle reports their Pearson correlation and slope.
Per-gene real and synthetic log2FC, adjusted p-values and DE calls for every tested
gene are saved as one CSV per contrast (Table 7).

#if figures.differential_expression.len() > 0 {
  figure(
    panel-grid(figures.differential_expression.map(path => image(path, width: 100%))),
    caption: [*Effect-size concordance of differential expression in the full cohorts.* One panel per contrast, in the order of Table 3. DESeq2 was fitted separately to the full real and synthetic cohorts on #if cohort.de_all_genes [all #(cohort.de_genes_tested) genes] else [the #(cohort.de_genes_tested) most variable genes]; the #(cohort.de_genes_reported) genes in the top 80% by real mean CPM are shown. Each point is a real DE gene (real padj < 0.05 and |log2FC| > 0.5) at its real (x) and synthetic (y) log2 fold change, first-named group relative to the second (positive = higher in the first-named group, e.g. higher in 0 for "0 vs 1"). Dashed line, identity: points on it mean real effects are reproduced in direction and size. Solid line, least-squares fit; its slope (subtitle) below 1 means synthetic effects are shrunk toward zero. Subtitles give Pearson r (at least 0.8 passes) and the slope. Non-anchor feature marks a column not supplied to dbTwin as an anchor.],
  )
}

#if figures.gene_counts.len() > 0 {
  figure(
    panel-grid(figures.gene_counts.map(path => plot-panel(path, width: 100%))),
    caption: [*Recall of real DE genes in the full cohorts.* One panel per contrast, in the order of Table 3. Black: real DE genes (padj < 0.05 and |log2FC| > 0.5). Green pairs, at strict (padj < 0.05, |log2FC| > 0.5; darker) and relaxed (padj < 0.075, |log2FC| > 0.4; lighter) synthetic thresholds: all synthetic DE genes (solid) and those that are also real DE genes (hatched, recalled). Percentages give recall, the share of real DE genes recalled; precision is in Table 3. A solid bar well above its hatched bar indicates synthetic-only discoveries. The relaxed threshold is recommended for small subgroups; recall of at least 80% under this threshold passes.],
  )
}

== ML fidelity: train-on-synthetic, test-on-real (TSTR) AUC and SHAP concordance

Predictive utility was evaluated using train-on-synthetic, test-on-real (TSTR)
analysis. Elastic-net logistic regression classifiers trained on each synthetic cohort
were evaluated on the corresponding held-out real fold. Train-on-real,
test-on-real (TRTR) models used the same held-out subjects and provide the
real-data reference. Mean ROC-AUC was #strong(metric("roc_auc_trtr").estimate) for TRTR
and #strong(metric("roc_auc_tstr").estimate) for TSTR (mean ± SD over outer splits;
each split is shown in the figure below). The comparison therefore tests
out-of-sample analytical utility rather than the ability to fit synthetic
records themselves.

#if figures.auc_folds != none {
  figure(
    plot-panel(figures.auc_folds, width: 55%),
    caption: [*TRTR and TSTR ROC-AUC in each outer split.* Each pair of dots is one split: the real-trained (dark grey) and synthetic-trained (green) elastic-net classifier for #raw(ml-target), both scored on the same held-out real fold. Dashed lines give the means reported above. TSTR dots level with their TRTR partners mean synthetic-trained models generalize to new real patients as well as real-trained ones.],
  )
}

Matching AUC is not sufficient for transcriptomic discovery: two classifiers
can discriminate the same classes while relying on different genes. We
therefore compared mean absolute SHAP values from real-trained and
synthetic-trained classifiers, with both explanations evaluated on held-out
real samples. Correlations were computed for the 50 genes with the largest
real-trained mean absolute SHAP values. Class-specific results are reported
below because averaging across classes can obscure minority-class failures.
Synthetic SHAP signal was absent in #data.shap_failed_folds of #data.shap_total_folds
outer folds despite variable real SHAP signal.

#v(8pt)
#block(breakable: false, width: 100%)[
#if not (ml-target in anchor-cols) [#non-anchor-label() #v(4pt)]
#table-caption(4, [Class-specific concordance of real-trained and synthetic-trained SHAP gene attributions; correlations are mean ± SD over valid splits. Top-10 genes shared: how many of the ten genes with the highest real-trained mean |SHAP| are also among the synthetic-trained top ten, averaged over all splits.])
#report-table(
  ("ML target class", "SHAP Pearson r", "SHAP Spearman rho", "Top-10 genes shared", "Splits"),
  shap-results.map(item => (
    item.class, strong(item.pearson.estimate), item.spearman.estimate, item.top10,
    item.pearson.splits,
  )),
  column-widths: (auto,) * 5, width: auto,
)
]

#v(10pt)

These correlations test concordance of the evaluated classifier's feature
attributions, not causal gene effects. They are nevertheless a stringent check
that synthetic-trained models recover the same predictive signal when applied
to unseen real subjects. The plots show the nine leading real-trained genes for
a valid split in which at least one plotted gene has non-zero real and
synthetic SHAP values for every class; annotations summarize correlations as mean ± SD across valid outer
splits.
#if data.shap_split == none [No split had valid SHAP correlations and non-zero plotted SHAP values for every class, so SHAP panels were omitted.]

#for start in range(0, figures.shap.len(), step: 3) {
  let panels = figures.shap.slice(start, calc.min(start + 3, figures.shap.len()))
  figure(
    grid(
      columns: panels.len(), gutter: 8pt,
      ..panels.map(path => [
        #block(
          image(path, width: 100%, height: 100%, fit: "contain"),
          width: 100%, height: 2in,
        )
        #if not (ml-target in anchor-cols) [\ #non-anchor-label()]
      ]),
    ),
    caption: [*Gene-level drivers of real- and synthetic-trained classifiers, representative split #data.shap_split.* Elastic-net models for #raw(ml-target) were trained on the real training fold (dark grey) or the matching synthetic fold (green), using the same genes and regularization, and both were explained on the same held-out real subjects. One panel per class (one for binary targets). Bars show mean absolute SHAP values for the nine genes with the highest real-trained importance in this split; similar heights mean the synthetic-trained model relies on the same genes. Titles give the Pearson r over the top 50 real genes as mean ± SD across valid splits (Table 4).],
  )
}

== Empirical privacy evaluation

Privacy was evaluated using two distinct empirical tests. DCR measures
memorization-like proximity in expression space. Each synthetic sample's distance
to its nearest real training sample was compared with the reference distribution
of distances from each real training sample to its nearest other training sample.
The ratio of the fifth percentiles of these distributions was
#metric("privacy_p5_ratio").estimate. Ratios well below 1 suggest unusually close
synthetic records relative to the spacing between distinct training records;
this is an empirical memorization check, not a privacy guarantee.

Attribute inference addresses a different threat model: the attacker has the synthetic data and some features of a real patient, called auxiliary columns, and tries to guess that patient's unknown features, called secrets. To separate leakage from genuine population-level biology, the same attack is repeated against held-out real patients who were not used for generation (the control). Attribute inference is a concern when the training attack success rate is substantially higher than the control success rate.

We tested two threat models. In the first, the attacker knows only the clinical quasi-identifiers (#raw(cohort.qi_cols)); in the second, the attacker also knows the top #(cohort.n_genes_aux) most variable genes, ranked by variance of log2(CPM + 1). In both attacks the secrets are the remaining clinical columns and the next #(cohort.n_genes_secret) most variable genes, and each secret column was attacked independently. The reported statistic is training minus control attack success [2], averaged across all available secret-by-split evaluations. All attribute inference attacks were implemented using the Anonymeter package [3].

#v(8pt)
#block(breakable: false, width: 100%)[
#table-caption(5, [Summary of privacy measures; mean ± SD across outer splits or attribute-inference evaluations.])
#report-table(
  ("Metric", "Estimate"),
  (
    ("DCR 5th percentile ratio", metric("privacy_p5_ratio").estimate),
    ("Attribute inference: expression + clinical", metric("attribute_inference_standard").estimate),
    ("Attribute inference: clinical only", metric("attribute_inference_clinical_qi").estimate),
  ),
  column-widths: (auto, auto), width: auto,
)
]

Feature counts in Table 5 include all secret columns whose mean training minus control
success across splits is strictly above 0.2, separately for each attack. Excess alone
does not show how successful an attack is, so Table 6 gives the absolute success rates
behind the largest excesses.

#v(6pt)
#block(breakable: false, width: 100%)[
#table-caption(6, [Secrets with the five largest mean excesses for each attack. Attack success rates are averaged over outer splits: training (attacking patients used for generation) and control (held-out patients). Excess is training minus control.])
#report-table(
  ("Attack", "Secret", "Training", "Control", "Excess"),
  data.top_secrets.map(item => (
    item.attack, raw(item.secret), item.training, item.control, strong(item.excess),
  )),
  column-widths: (auto,) * 5, width: auto, inset: 4pt,
)
]


#if figures.privacy != none {
  figure(
    image(figures.privacy, width: 100%),
    caption: [*Empirical privacy tests.* (A) Distance to closest record (DCR), pooled over outer splits: each synthetic record to its nearest real training record (left) versus each training record to its nearest other training record (right), on standardized log2(CPM + 1) expression. Copied or lightly perturbed records would pull the left violin toward zero; the annotation gives the 5th-percentile ratio (mean ± SD across splits), where values near or above 1 show no excess proximity. (B) Attribute inference (Anonymeter): an attacker who knows #raw(cohort.qi_cols) alone, or these plus #(cohort.n_genes_aux) expression genes, infers a withheld clinical or gene secret from the closest synthetic record. The y-axis is success on training patients minus success on held-out patients, isolating what the synthetic data reveals about the individuals used to build it. The seven secrets with the highest mean excess are shown; boxes span splits, dots are splits, and the axis starts at 0.],
  )
}


The two attack estimates describe training-specific advantage under the
implemented auxiliary information and warrant consideration in release
decisions. Importantly, a high attribute inference score does not automatically raise re-identification risk: that also depends on the sensitivity of the attacked secret and of the patient population [2]. DCR tests close
record replication, whereas attribute inference tests attack advantage; the two results are not interchangeable.

= How to read and use this report (FAQs)

*What do Pass and Fail mean in Table 2?* Each metric is compared with a fixed
threshold: co-expression rho at least 0.8; UMAP MMD at most 0.1; TSTR AUC
within 0.05 of TRTR AUC; DE log2FC r at least 0.8; DE gene recall (real DE genes recalled
at the relaxed synthetic threshold: padj < 0.075 and |log2FC| > 0.4) at least 80%; SHAP Pearson r at
least 0.8.
Non-anchor columns are exempt from the thresholds and are omitted from Table 2; their
DE contrasts are listed as not graded in the Bottom line.

*How do I interpret the co-expression matrix?* Real gene--gene correlations
are shown below the diagonal and synthetic correlations above it, in the same
gene order. A Spearman rho close to 1 means the synthetic data has the same
gene correlation structure as the real data.

*What does the UMAP MMD mean?* MMD (maximum mean discrepancy) measures the separation between real and
synthetic samples in the joint UMAP embedding. An MMD of 0 means real and
synthetic samples are indistinguishable in that projection; larger values
indicate increasing distance. 

*How should I compare TSTR and TRTR?* TRTR is the real-data reference. TSTR
close to TRTR means models trained on synthetic data achieve similar AUC on
held-out real subjects as models trained on real data.

*What does the DE log2FC r show?* It correlates real and synthetic effect sizes
for all real DE genes (real padj < 0.05 and |log2FC| > 0.5). A value close to 1 means the
synthetic cohort reproduces the same direction and ordering of effects. The slope adds
magnitude: near 1, synthetic effects keep their real size; below 1, they are shrunk
toward zero even when r is high.

*How do I read the DE gene counts?* Recall is the share of real DE genes that
are also significant in the synthetic cohort (recalled count in Table 3); Total is all synthetic DE genes.
A recalled count close to the real DE count means real discoveries are recovered, while
Total far above it indicates synthetic-only discoveries. Precision is the
recalled count divided by Total: the share of synthetic DE genes that are also real DE
genes.

*What does SHAP concordance show?* A SHAP r close to 1 means synthetic-trained
classifiers rely on the same genes, with similar relative importance, as
real-trained classifiers.

*How do I read the privacy results?* The DCR ratio divides the fifth percentile
of synthetic-to-training nearest-neighbor distances by the fifth percentile of
training-to-other-training nearest-neighbor distances. Ratios near 1 indicate
comparable lower-tail distances; much lower ratios suggest memorization.
Attribute-inference excess
success near 0 means the synthetic data gives an attacker no advantage beyond
population-level knowledge.

*How should I use the synthetic cohort?* For exploratory analysis, hypothesis
generation, workflow development, and preliminary feature prioritization,
followed by confirmation on governed real data (see Limitations).

= Methods

Clinical fidelity, differential expression, UMAP, and co-expression are evaluated once using the full real and synthetic cohorts. These comparisons are in-sample: the full synthetic release was generated from every real sample, and it is compared against those same samples. Predictive ML accuracy, SHAP concordance, DCR and attribute inference are evaluated using a 5-fold train-test split with synthetic datasets generated only from the training folds, so ML accuracy, SHAP and the control arm of attribute inference use real samples held out from generation. All synthetic folds are saved in #raw("datasets/<data-folder>/dbTwin/artifacts/") and only used the training folds as delineated in the split_manifest. 

All real data is stored in #raw("datasets/<data-folder>/real_data").

== Global and clinical fidelity

Clinical anchor-column distributions were compared within ML-target classes
in the full real and synthetic cohorts. As a sampling-noise reference, each real class
was compared with 500 bootstrap resamples of itself, each the size of the matching
synthetic class; an observed TVD at or below the 97.5th percentile of these values is
within sampling noise. Each column was treated as categorical or continuous once, from
its values in the full real cohort, for every class and for the noise reference.
Expression analyses used log2(CPM + 1), with counts per million (CPM) from each sample's
total counts over all genes. For co-expression, genes were ranked by variance of these values in
the full real cohort, and Pearson gene--gene correlation matrices of the top 500 genes
were computed separately in real and synthetic data. Average-linkage
hierarchical ordering derived from the real matrix was applied to both.
Spearman correlation between corresponding upper-triangle entries provided the
scalar co-expression statistic.

For UMAP, the full real and synthetic cohorts were transformed to log2(CPM + 1), so that
sequencing depth does not drive the embedding, and genes at or below the 40th percentile
of full-real-cohort variance were excluded. The concatenated cohorts were reduced to 100
principal components (centred, not scaled) and embedded in a single two-dimensional UMAP
fit with 15 neighbors, minimum distance 0.1, Euclidean distance, and random state 42. MMD
was computed with an RBF kernel on these two-dimensional coordinates. These analyses were
computed once for the full cohorts.

== Differential expression

#if cohort.de_all_genes [All #(cohort.de_genes_tested) genes were tested.] else [The
#(cohort.de_genes_tested) genes with highest log2(CPM + 1) variance in the full real cohort were
selected once.] DESeq2 models were then fit independently to rounded
integer counts from the full real and synthetic cohorts using each
contrast's specified clinical column as the design factor. As in R DESeq2, counts flagged
as outliers by Cook's distance were replaced and those genes refit (levels with at least
seven samples); log2 fold-changes were not shrunk.
Adjusted p-values and log2 fold-changes were aligned by gene.
Real DE genes were those with adjusted p-value below 0.05 and |log2FC| above
0.5 in the real cohort; the Pearson correlation and least-squares slope of
synthetic on real log2 fold-change used all of them. log2 fold-changes are the
first-named group relative to the second. Synthetic DE genes use two thresholds:

- *Strict:* padj < 0.05 and |log2FC| > 0.5.
- *Relaxed:* padj < 0.075 and |log2FC| > 0.4.

Table 3 reports total synthetic DE genes, recall of the real DE set
(recalled count and percentage), and precision (recalled count divided by total). Each scatter plot
shows one contrast in the full cohorts, with its Pearson r and slope in the subtitle.

== Machine learning and SHAP

The classifier was elastic-net logistic regression with balanced class weights
and the SAGA solver. In each outer training cohort, the number of expression
features was the minimum of 270, 90% of the training sample count, and the
available gene count. Counts were transformed to log2(CPM + 1), genes were ranked by
training-set variance of these values, missing values were median-imputed, and features
were standardized. The SAGA solver used random state 42.

Regularization was selected using only real training data. Within every outer
training cohort, stratified cross-validation used up to five folds, limited by
the smallest class. The l1 ratio was fixed at 0.05 and $C$ was evaluated at 51
log-spaced values from $10^(-4)$ to $10^1$. A candidate was eligible only when
the active-feature fraction lay between 20% and 60% in every inner fold of
every outer split. The eligible value with the highest fold-weighted mean AUC
was selected, with smaller $C$ preferred on ties, and then applied unchanged to
real- and synthetic-trained models.

TRTR and TSTR classifiers were evaluated on the same held-out real subjects.
For multiclass ML targets, the reported ROC-AUC is the mean of one-versus-rest
class AUCs. Linear SHAP values were computed on the standardized held-out real
features. Within each class and split, mean absolute SHAP values were compared
between real- and synthetic-trained classifiers over the top 50 genes selected
by real-trained importance using Pearson and Spearman correlation. Correlations
were defined only when both selected importance vectors were finite and had
positive standard deviation; undefined folds were excluded from mean and SD.
Signal failures counted unique folds where any class had variable real top-50
importance but all-zero synthetic importance.

== Privacy analyses

For DCR, all genes were transformed to log2(CPM + 1), missing values were set to
zero, and genes were standardized using the real training mean and standard
deviation. Each synthetic sample's Euclidean distance to its nearest real
training sample was computed. The reference distribution consisted of each
real training sample's Euclidean distance to its nearest other real training
sample, excluding the sample itself. For each split, the fifth percentile of
the synthetic-to-training distances was divided by the fifth percentile of the
real-training reference distances. The report gives the mean and standard
deviation of these ratios across splits. Held-out samples and reference-set
subsampling were not used in this DCR calculation.

Attribute inference used Anonymeter's inference evaluator at 95% confidence,
with one attack per held-out subject. The configured clinical
quasi-identifiers were #raw(cohort.qi_cols). Eligible clinical secrets were
#raw(cohort.clinical_secrets). Genes were ranked by log2(CPM + 1) variance in each
fold's synthetic cohort. Gene secrets were the next #(cohort.n_genes_secret) ranked
genes after the #(cohort.n_genes_aux) auxiliary genes, with no overlap, and kept their
released raw counts. The expression-assisted attack used the auxiliary genes as
log2(CPM + 1) standardized with real training statistics; the second attack used
clinical quasi-identifiers alone. The
reported excess success is the training attack rate minus the control attack
rate, averaged over available secret-by-split evaluations for each attack. This
is distinct from Anonymeter's normalized inference-risk estimator.

= Data files and layout

#let files = data.data_files
#let file-path(name) = raw(name)
Paths in Table 7 are relative to #raw(files.root + "/") in the benchmark repository. On
Hugging Face (NandaAditya/rna-twin) the synthetic CSVs sit under #raw(files.cohort + "/")
with the same names and #raw("artifacts/") subfolder.

#v(4pt)
#block(breakable: false, width: 100%)[
#table-caption(7, [Released and benchmark files for #(cohort.name).])
#report-table(
  ("File", "Contents"),
  (
    (file-path(files.release), [*Full synthetic release* (#files.release_rows samples), generated once from all #(cohort.real_samples) real samples. This is the cohort intended for reuse.]),
    (file-path("artifacts/" + files.fold_pattern), [Benchmark-only synthetic cohorts, one per outer fold, each generated from that fold's real training samples (TSTR, SHAP, DCR, attribute inference).]),
    (file-path("../real_data/split_manifest.csv"), [Outer-fold assignment of real samples: #raw("sample_id"), #raw("split"), #raw("partition") (train or test).]),
    (file-path("bench_metrics.yaml"), [ML target, DE contrasts, anchor and quasi-identifier columns and gene counts; a reusable template for defining custom DE/ML analyses.]),
    (stack(spacing: 3pt, ..files.de_files.map(file-path)), [Per-gene DE results, one file per contrast in Table 3 order: #raw("gene"), #raw("gene_symbol"), real and synthetic #raw("log2fc") and #raw("padj"), #raw("passes_expression_filter") (top 80% by real mean CPM), and #raw("real_de"), #raw("synthetic_de_strict"), #raw("synthetic_de_relaxed") flags as counted in Table 3.]),
    (file-path("artifacts/metrics.csv"), [Summary metrics. Per-secret, per-record and per-class results are in #raw("attribute_inference.csv"), #raw("dcr.csv") and #raw("clinical_fidelity_metrics.csv") in the same folder.]),
    (raw("../real_data/"), [Real #raw("expression.csv.gz") (samples × genes) and #raw("metadata.csv") (clinical variables by sample ID), where the source permits redistribution.]),
  ),
  column-widths: (auto, 1fr), cell-align: left + horizon, inset: 5pt, justify: false,
)
]

- *Layout.* Each synthetic CSV has one row per synthetic sample. The first, unnamed
  column is a row number, not a real sample ID: synthetic samples do not correspond to
  individual real patients. The remaining columns are the clinical variables
  (#files.clinical_columns.map(raw).join(", ")) and #files.n_genes gene columns.
  #if files.release_only.len() > 0 [The full release also contains #files.release_only.map(raw).join(", "), which the fold files do not.]
  #if files.fold_only.len() > 0 [The fold files also contain #files.fold_only.map(raw).join(", "), which the full release does not.]
- *Counts.* Gene values are raw, unnormalized, non-negative integer counts: no
  library-size normalization or log transform has been applied. Normalize them as any
  RNA-seq count matrix (e.g. DESeq2 size factors or log CPM).
- *Gene IDs.* Gene columns keep the source study's identifiers: #files.n_ensembl Ensembl
  gene IDs (e.g. #raw(files.ensembl_example), #if files.versioned [with] else [without] version suffixes)#if files.n_symbols != "0" [ and #files.n_symbols gene symbols (e.g. #files.symbol_examples.map(raw).join(" and ")), as distributed by the source; some may be retired HGNC symbols]. The #raw("gene_symbol") column of the DE files maps Ensembl IDs with a packaged HGNC table; unmapped names are kept as they are.

#v(18pt)
= Limitations and intended use

This report supports a dataset-specific assessment of dbTwin under the analyses
implemented above. Several limitations define its proper interpretation.

- Successful benchmarking does not imply fidelity for analyses that were not
  tested. In particular, this pipeline does not evaluate pathway enrichment,
  survival models, network inference, rare-variant or tail behavior, or
  replication in an external cohort.

- DE results are limited to the specified contrasts and #if cohort.de_all_genes [the
  #(cohort.de_genes_reported) most highly expressed genes] else [the
  #(cohort.de_genes_reported) most highly expressed of the #(cohort.de_genes_tested)
  most variable genes]. ML results use one classifier family and a target-focused
  feature-selection scheme. Other estimands, preprocessing choices, and model
  classes may behave differently.

- The five outer folds are deterministic and stratified by the primary ML target.
  Minority classes may still produce unstable fold-specific estimates. SHAP figures
  display one representative split; DE, UMAP, clinical fidelity
  and co-expression use the full cohorts once.

- UMAP and MMD quantify agreement after a particular nonlinear projection;
  co-expression is restricted to 500 variable genes. These summaries do not
  fully characterize the high-dimensional joint distribution.

- DCR is a memorization test, not a privacy proof. Attribute inference is an
  empirical, threat-model-specific attack using the stated auxiliary variables
  and evaluator. Neither analysis provides differential privacy or guarantees
  protection against linkage, genotype reconstruction, or attacks with
  different external information.

The synthetic cohort is therefore best suited to exploratory analysis,
hypothesis generation, workflow development, and preliminary feature
prioritization, followed by confirmation on governed real data.
It should not be treated as a substitute for real-cohort validation in clinical,
regulatory, or other high-stakes settings.

= References

[1] Nanda A, Saha S. rna-twin: synthetic bulk RNA-seq cohorts and benchmark (2026). #link("https://github.com/Nanda-Aditya/rna-twin")

[2] Pilgram L, Dankar FK, Drechsler J, Elliot M, Domingo-Ferrer J, Francis P, Kantarcioglu M, Kong L, Malin B, Muralidhar K, Myles P, Prasser F, Raisaro JL, Yan C, El Emam K. A consensus privacy metrics framework for synthetic data.

[3] Giomi M, Boenisch F, Wehmeyer C, Tasnádi B. A unified framework for quantifying privacy risk in synthetic data. _Proceedings on Privacy Enhancing Technologies_ 2023(2). #link("https://petsymposium.org/popets/2023/popets-2023-0055.php")

#if figures.gene_stats != none [
  #pagebreak(weak: true)
  = Appendix A. Per-gene statistics and library size

  These standard RNA-seq checks compare the full real cohort with the full synthetic
  release, so, like the other full-cohort comparisons, they are in-sample.

  #counter(figure.where(kind: image)).update(0)
  #set figure(numbering: n => "A" + str(n))
  #figure(
    image(figures.gene_stats, width: 85%),
    caption: [*Per-gene statistics and library size, full real cohort versus full synthetic release.* Per-gene statistics use log2 counts per million plus one (CPM from each sample's total counts) for all #(cohort.genes) genes. (A) Mean of each gene, real (x) versus synthetic (y). (B) Variance of each gene across samples; the title also gives the median synthetic-to-real variance ratio over genes with non-zero real variance (1 = variance kept, below 1 = smoothed). (C) Fraction of samples with a zero count. Dashed lines, identity; titles give the Pearson r over genes. (D) Library size (total counts per sample) as violins with box plots (median, IQR, whiskers to 1.5 IQR).],
  )
]
