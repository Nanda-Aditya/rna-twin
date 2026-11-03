"""Prepare the Anonymeter runtime and download the gated synthetic RNA-seq cohorts.

Run from the repository root: uv run python get_data.py
(uv builds .venv from uv.lock first; .python-version pins Python 3.12.)
"""

from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import NamedTemporaryFile
import webbrowser

from huggingface_hub import HfApi, hf_hub_download, login
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

root = Path(__file__).resolve().parent
subprocess.run([sys.executable, str(root / "scripts/setup_anonymeter.py")], cwd=root, check=True)

repo = "NandaAditya/rna-twin"
url = f"https://huggingface.co/datasets/{repo}"
missing = []
for cohort in ("pediatric_risk", "sepsis", "tcga_luad"):
    output = root / "datasets" / cohort / "dbTwin"
    names = [f"syn_{cohort}.csv", *(
        f"artifacts/syn_{cohort}_fold_{fold}.csv" for fold in range(1, 6))]
    missing.extend((f"{cohort}/{name}", output / name, output)
                   for name in names if not (output / name).is_file())

if missing:
    try:
        login()
        api = HfApi()
        try:
            api.auth_check(repo, repo_type="dataset")
        except GatedRepoError:
            print(f"Read and accept the DUA using the same HF account: {url}")
            webbrowser.open(url)
            input("After accepting the DUA in your browser, press Enter to continue: ")
            api.auth_check(repo, repo_type="dataset")
        revision = api.repo_info(repo, repo_type="dataset", revision="v1.0.0").sha
        for filename, destination, output in missing:
            print(f"Downloading {filename}", flush=True)
            cached = hf_hub_download(repo, filename, repo_type="dataset",
                                     revision=revision, token=True)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with (Path(cached).open("rb") as source,
                  NamedTemporaryFile(dir=output, prefix="syn_download_", suffix=".csv",
                                     delete=False) as staged):
                shutil.copyfileobj(source, staged)
            Path(staged.name).replace(destination)
    except GatedRepoError as error:
        raise SystemExit(
            f"Access is still gated.\n 1. Open {url}, log in, and accept the DUA.\n"
            " 2. Run 'uv run hf auth login' with the same account.\n"
            " 3. Rerun get_data.py."
        ) from error
    except HfHubHTTPError as error:
        raise SystemExit(
            f"HF access/download failed: {error}\nCheck your HF account and token permissions. "
            "If the repository is private, the owner must grant access or release it with gating."
        ) from error
    except OSError as error:
        raise SystemExit(f"Dataset download failed: {error}. Rerun get_data.py to resume.") from error
else:
    print("All 18 synthetic CSVs are already present; no HF download needed.")

print("Ready: .venv and .venv_anonymeter are set up and all synthetic datasets are present.")
print("Run the benchmark with: uv run python scripts/run_pipe.py")
