"""Install the isolated Anonymeter runtime after uv syncs the main project."""

import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
runtime = root / ".venv_anonymeter"
python = runtime / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")

if not python.is_file():
    subprocess.run(["uv", "venv", str(runtime), "--python", "3.12"], cwd=root, check=True)
subprocess.run([
    "uv", "pip", "install", "--python", str(python),
    "-r", str(root / "requirements_anon.txt"),
], cwd=root, check=True)
print(f"Anonymeter runtime ready: {python}")