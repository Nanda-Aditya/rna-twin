#!/usr/bin/env python3
"""Run the benchmark for one or all cohorts: metrics, figures, and the PDF report."""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

scripts = Path(__file__).resolve().parent

def run_pipe(data, full_compute=True):
    # Every step reads the split manifest from input_dir (real_data/split_manifest.csv).
    input_dir = scripts.parent / "datasets" / data / "real_data"
    output_dir = scripts.parent / "datasets" / data / "dbTwin"
    artifacts_dir = output_dir / "artifacts"
    bench_config_path = output_dir / "bench_metrics.yaml"
    with bench_config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    metrics_files = (
        "metrics.csv", "attribute_inference.csv", "dcr.csv",
        "clinical_fidelity_metrics.csv", "clinical_fidelity_data.csv",
        "benchmark_artifacts.npz")
    commands = []
    if full_compute:
        commands.append((
            "compute_metrics.py", input_dir, artifacts_dir,
            "--ml-target", config["ml_target"],
            "--anchor-cols", config["anchor_cols"],
            "--qi-cols", *config["qi_cols"],
            "--n-genes-aux", config["n_genes_aux"],
            "--n-genes-secret", config["n_genes_secret"],
            "--n-genes-de", config["n_genes_de"],
            *(arg for contrast in config["de_contrasts"] for arg in (
                "--contrast", contrast["column"], *contrast["categories"]))))
    elif all((artifacts_dir / name).is_file() for name in metrics_files):
        print(f"Reusing saved metrics in {artifacts_dir}")
    else:
        raise SystemExit(f"No saved metrics in {artifacts_dir}; run without --no-full-compute.")
    commands.extend((
        ("create_figs.py", artifacts_dir, "--bench-config", bench_config_path),
        ("create_report.py", output_dir, input_dir,
         "--cohort-name", config["cohort_name"], "--bench-config", bench_config_path)))
    for script, *arguments in commands:
        subprocess.run(
            [str(arg) for arg in (sys.executable, scripts / script, *arguments)],
            check=True)


def main():
    cohorts = ("pediatric_risk", "sepsis", "tcga_luad")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", choices=cohorts,
                        help="Cohort to evaluate; omit to run all three.")
    parser.add_argument("--no-full-compute", action="store_true",
                        help="Reuse saved metrics and only rebuild figures and the report "
                             "(minutes instead of hours).")
    args = parser.parse_args()
    datasets = [args.dataset] if args.dataset else cohorts
    for data in datasets:
        run_pipe(data, not args.no_full_compute)


if __name__ == "__main__":
    main()
