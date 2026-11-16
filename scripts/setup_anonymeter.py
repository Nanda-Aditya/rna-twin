"""Install the isolated Anonymeter runtime after uv syncs the main project, then check it."""

import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from bench_utils.attr_infer import _runtime_environment, anonymeter_python  # noqa: E402

runtime = root / ".venv_anonymeter"
python = runtime / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")

if not python.is_file():
    subprocess.run(["uv", "venv", str(runtime), "--python", "3.12"], cwd=root, check=True)
subprocess.run([
    "uv", "pip", "install", "--python", str(python),
    "-r", str(root / "requirements_anon.txt"),
], cwd=root, check=True)

# Start the interpreter exactly as the benchmark does and confirm it runs inside the venv
# with NumPy 1.x and Anonymeter importable.
worker = anonymeter_python()
check = subprocess.run(
    [str(worker), "-c",
     "import sys, numpy, anonymeter; print(sys.prefix); print(numpy.__version__)"],
    cwd=root, env=_runtime_environment(), capture_output=True, text=True, check=False,
)
lines = check.stdout.split()
problem = None
if check.returncode:
    problem = check.stderr.strip() or f"exit code {check.returncode}"
elif len(lines) != 2:
    problem = f"unexpected output: {check.stdout.strip()!r}"
elif Path(lines[0]).resolve() != runtime.resolve():
    problem = f"it runs outside the venv (sys.prefix={lines[0]})"
elif not lines[1].startswith("1."):
    problem = f"it has NumPy {lines[1]}; Anonymeter needs NumPy 1.x"
if problem:
    raise SystemExit(
        f"Anonymeter runtime check failed for {worker}: {problem}\n"
        f"Delete {runtime} and rerun: uv run python scripts/setup_anonymeter.py"
    )
print(f"Anonymeter runtime ready: {worker} (NumPy {lines[1]})")
