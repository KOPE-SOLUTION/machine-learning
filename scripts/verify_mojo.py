"""Compile both Mojo lessons and compare outputs with their Python references."""
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/verification/mojo"
OUT.mkdir(parents=True, exist_ok=True)
MOJO = Path(sys.executable).parent / "mojo"
if not MOJO.is_file():
    raise SystemExit("Run with .venv-mojo/bin/python after installing mojo/requirements.lock")
env = os.environ.copy()
env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
env["PATH"] = str(MOJO.parent) + os.pathsep + env.get("PATH", "")
version = subprocess.check_output([str(MOJO), "--version"], text=True, env=env).strip()
if not version.startswith("Mojo 1.2.0"):
    raise SystemExit(f"Expected Mojo 1.2.0, found {version}")
rows = []
for slug in ("01-gradient-descent", "02-python-interop"):
    outputs = {}
    for kind, executable, filename in (("mojo", MOJO, "main.mojo"),
                                       ("python", Path(sys.executable), "reference.py")):
        result = subprocess.run([str(executable), str(ROOT / "mojo" / slug / filename)],
                                cwd=ROOT, env=env, text=True, encoding="utf-8",
                                capture_output=True, timeout=180)
        (OUT / f"{slug}-{kind}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        result.check_returncode()
        outputs[kind] = result.stdout
    if slug == "01-gradient-descent":
        values = {kind: [float(line.split(":", 1)[1]) for line in output.splitlines()]
                  for kind, output in outputs.items()}
        np.testing.assert_allclose(values["mojo"][:2], [3, 2], rtol=0, atol=1e-8)
        np.testing.assert_allclose(values["mojo"], values["python"], rtol=1e-10, atol=1e-12)
        assert 0 <= values["mojo"][2] < 1e-12
    else:
        values = {}
        for kind, output in outputs.items():
            accuracy = float(re.search(r"test accuracy:\s*([0-9.]+)", output)[1])
            predictions = np.fromstring(re.search(r"predictions:\s*\[([^]]+)\]", output)[1],
                                        dtype=int, sep=" ")
            values[kind] = (accuracy, predictions)
        np.testing.assert_allclose(values["mojo"][0], values["python"][0], rtol=0, atol=1e-12)
        np.testing.assert_array_equal(values["mojo"][1], values["python"][1])
        np.testing.assert_array_equal(values["mojo"][1], [0, 2, 1, 1, 0])
    rows.append({"lesson": slug, "passed": True, "outputs": outputs})
    print(f"PASS Mojo and Python agree: {slug}", flush=True)
report = {"mojo": version, "python": platform.python_version(),
          "platform": platform.platform(), "results": rows}
(OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
