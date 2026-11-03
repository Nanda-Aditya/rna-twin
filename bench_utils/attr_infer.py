"""Compute Anonymeter attribute-inference risk in an isolated environment."""

from __future__ import annotations

import os
from pathlib import Path
import pickle
import subprocess
import sys
from tempfile import TemporaryDirectory


CONFIDENCE = 0.95
N_JOBS = -4
ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ENV = "RNA_TWIN_ANONYMETER_RUNTIME"


def in_anonymeter_runtime():
    """Whether this process is the Anonymeter worker started by `compute_secret_risk`."""
    return os.environ.get(RUNTIME_ENV) == "1"


def anonymeter_python():
    """Path of the isolated Anonymeter venv's Python (NumPy 1.x), made by setup_anonymeter.py."""
    python = (ROOT / ".venv_anonymeter" /
              ("Scripts/python.exe" if os.name == "nt" else "bin/python")).resolve()
    if not python.is_file():
        raise FileNotFoundError(
            f"Anonymeter Python was not found at {python}. Run "
            "'uv run --python 3.12 python scripts/setup_anonymeter.py' "
            "from the rna-twin repo root."
        )
    return python


def _runtime_environment():
    """Current environment, marked as the Anonymeter runtime, with this repository importable."""
    env = os.environ.copy()
    env[RUNTIME_ENV] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(ROOT), env.get("PYTHONPATH")) if part
    )
    return env


def compute_secret_risk(
    df_trn,
    df_syn,
    df_tst,
    genes,
    auxiliary,
    secret,
    seed,
):
    """Anonymeter attribute-inference risk for one `secret` column.

    The attacker knows the auxiliary columns (`genes` + clinical `auxiliary`) of real records
    and uses `df_syn` to guess `secret`; success on training records is compared with held-out
    `df_tst` controls (outputs in `_compute_secret_risk`). The `genes` columns must already be
    log2(CPM + 1) (`expression.log_cpm` over all genes); a gene secret keeps its released
    scale. Runs in-process inside the Anonymeter runtime, otherwise in that venv through a
    pickled subprocess call.
    """
    if in_anonymeter_runtime():
        return _compute_secret_risk(
            df_trn, df_syn, df_tst, genes, auxiliary, secret, seed
        )

    columns = list(dict.fromkeys([*genes, *auxiliary, secret]))
    payload = {
        "df_trn": df_trn.loc[:, columns],
        "df_syn": df_syn.loc[:, columns],
        "df_tst": df_tst.loc[:, columns],
        "genes": list(genes),
        "auxiliary": list(auxiliary),
        "secret": secret,
        "seed": seed,
    }
    # Windows can briefly lock the fresh pickles (e.g. antivirus scans); a leftover temporary
    # file must not discard a finished attack.
    with TemporaryDirectory(prefix="rna_twin_attr_", ignore_cleanup_errors=True) as temp:
        input_path = Path(temp) / "input.pkl"
        output_path = Path(temp) / "output.pkl"
        with input_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        completed = subprocess.run(
            [
                str(anonymeter_python()), "-m", "bench_utils.attr_infer",
                str(input_path), str(output_path),
            ],
            cwd=ROOT,
            env=_runtime_environment(),
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode:
            details = "\n".join(
                text.strip() for text in (completed.stdout, completed.stderr)
                if text.strip()
            )
            suffix = f":\n{details}" if details else "."
            raise RuntimeError(
                f"Anonymeter worker exited with code {completed.returncode}{suffix}"
            )
        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)
        if not output_path.is_file():
            raise RuntimeError("Anonymeter worker did not produce a result.")
        with output_path.open("rb") as handle:
            return pickle.load(handle)


def _compute_secret_risk(
    df_trn,
    df_syn,
    df_tst,
    genes,
    auxiliary,
    secret,
    seed,
):
    """Run Anonymeter's InferenceEvaluator with one attack per test record.

    Gene columns are z-scored with training statistics; the secret is a regression
    target when numeric with more than two values. Returns training, baseline and control
    attack success rates with errors, and `inference_risk` = (training - control) /
    (1 - control) with its confidence interval.
    """
    import numpy as np
    import pandas as pd
    from anonymeter.evaluators import InferenceEvaluator

    from bench_utils.privacy_metrics import _prep_expression

    aux = [*genes, *auxiliary]
    cols = [*aux, secret]
    n = len(df_tst)

    clinical = [*auxiliary, secret]
    if genes:
        E_trn, estats = _prep_expression(df_trn[genes])
        E_syn, _ = _prep_expression(df_syn[genes], fit_stats=estats)
        E_tst, _ = _prep_expression(df_tst[genes], fit_stats=estats)
        ori = pd.concat(
            [pd.DataFrame(E_trn, index=df_trn.index, columns=genes), df_trn[clinical]],
            axis=1,
        )
        syn = pd.concat(
            [pd.DataFrame(E_syn, index=df_syn.index, columns=genes), df_syn[clinical]],
            axis=1,
        )
        control = pd.concat(
            [pd.DataFrame(E_tst, index=df_tst.index, columns=genes), df_tst[clinical]],
            axis=1,
        )
    else:
        ori, syn, control = (
            df.loc[:, clinical] for df in (df_trn, df_syn, df_tst)
        )

    np.random.seed(seed)
    s = df_trn[secret]
    regression = pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) > 2

    evaluator = InferenceEvaluator(
        ori=ori[cols].reset_index(drop=True),
        syn=syn[cols].reset_index(drop=True),
        control=control[cols].reset_index(drop=True),
        aux_cols=aux,
        secret=secret,
        regression=regression,
        n_attacks=n,
    )
    evaluator.evaluate(n_jobs=N_JOBS)

    results = evaluator.results(confidence_level=CONFIDENCE)
    risk = evaluator.risk(confidence_level=CONFIDENCE)
    r_baseline = evaluator.risk(
        confidence_level=CONFIDENCE, baseline=True
    )

    return {
        "secret": secret,
        "n_attacks": n,
        "gene_aux_cols": tuple(genes),
        "clinical_aux_cols": tuple(auxiliary),
        "training_attack_success_rate": float(results.attack_rate.value),
        "training_attack_error": float(results.attack_rate.error),
        "baseline_attack_success_rate": float(results.baseline_rate.value),
        "baseline_attack_error": float(results.baseline_rate.error),
        "control_attack_success_rate": float(results.control_rate.value),
        "control_attack_error": float(results.control_rate.error),
        "inference_risk": float(risk.value),
        "inference_risk_ci_low": float(risk.ci[0]),
        "inference_risk_ci_high": float(risk.ci[1]),
        "r_baseline": float(r_baseline.value),
        "r_baseline_ci_low": float(r_baseline.ci[0]),
        "r_baseline_ci_high": float(r_baseline.ci[1]),
        "confidence_level": CONFIDENCE,
    }


def _run_worker(input_path, output_path):
    """Worker entry point: unpickle `compute_secret_risk` arguments, pickle the result.

    Pickles from the NumPy 2 benchmark environment reference `numpy._core`, which is mapped to
    the worker's NumPy 1 `numpy.core`.
    """
    import numpy as np

    if np.__version__.split(".", 1)[0] != "1":
        raise RuntimeError(
            f"Anonymeter worker requires NumPy 1.x, found {np.__version__}."
        )
    class NumPyCompatUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("numpy._core"):
                module = module.replace("numpy._core", "numpy.core", 1)
            return super().find_class(module, name)

    with Path(input_path).open("rb") as handle:
        payload = NumPyCompatUnpickler(handle).load()
    result = _compute_secret_risk(**payload)
    with Path(output_path).open("wb") as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    if len(sys.argv) != 3 or not in_anonymeter_runtime():
        raise SystemExit("attr_infer worker requires input and output paths.")
    _run_worker(sys.argv[1], sys.argv[2])
