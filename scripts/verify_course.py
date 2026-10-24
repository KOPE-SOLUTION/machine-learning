"""Run every Python lesson or notebook in isolation, with bounded runtime."""
import argparse
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["scripts", "readmes", "notebooks", "all"], default="scripts")
    args = parser.parse_args()
    out = ROOT / "outputs" / "verification"
    out.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(MPLBACKEND="Agg", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUTF8="1")
    rows = []
    if args.mode in ("scripts", "all"):
        for path in sorted(ROOT.glob("lessons/*/lesson.py")):
            start = time.perf_counter()
            result = subprocess.run([sys.executable, str(path)], cwd=ROOT, env=environment,
                                    text=True, encoding="utf-8", capture_output=True, timeout=180)
            (out / f"{path.parent.name}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
            rows.append({"lesson": path.parent.name, "kind": "script", "passed": result.returncode == 0,
                         "seconds": round(time.perf_counter() - start, 2)})
            print(f"{'PASS' if result.returncode == 0 else 'FAIL'} script {path.parent.name}", flush=True)
    if args.mode in ("readmes", "all"):
        # Each lesson starts fresh; explicitly sequential README cells share state.
        readmes = [*sorted(ROOT.glob("lessons/*/README.md")),
                   *sorted(ROOT.glob("mojo/*/README.md")), ROOT / "docs/math-primer.md"]
        for path in readmes:
            readme_text = path.read_text(encoding="utf-8")
            blocks = re.findall(r"^```python\n(.*?)^```", readme_text,
                                flags=re.MULTILINE | re.DOTALL)
            if "<!-- CODE:CELLS -->" in readme_text and blocks:
                blocks = ["\n\n".join(blocks)]
            if not blocks:
                raise SystemExit(f"No Python block found in {path.relative_to(ROOT)}")
            for index, code in enumerate(blocks, start=1):
                start = time.perf_counter()
                name = path.parent.name if path.name == "README.md" else path.stem
                result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=environment,
                                        text=True, encoding="utf-8", capture_output=True, timeout=180)
                (out / f"{name}-readme-{index}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
                rows.append({"lesson": name, "kind": "readme", "block": index,
                             "passed": result.returncode == 0,
                             "seconds": round(time.perf_counter() - start, 2)})
                print(f"{'PASS' if result.returncode == 0 else 'FAIL'} README {name} block {index}", flush=True)
    if args.mode in ("notebooks", "all"):
        import nbformat
        from nbclient import NotebookClient
        from jupyter_client.kernelspec import KernelSpecManager

        # Install a temporary kernelspec using this interpreter, not the system Python.
        kernel_dir = out / "kernels" / "course-python"
        kernel_dir.mkdir(parents=True, exist_ok=True)
        (kernel_dir / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Course verification", "language": "python", "env": {
                key: environment[key] for key in ("MPLBACKEND", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONUTF8")
            },
        }), encoding="utf-8")
        manager = KernelSpecManager(kernel_dirs=[str(kernel_dir.parent)])
        for path in sorted(ROOT.glob("lessons/*/lesson.ipynb")):
            start = time.perf_counter()
            nb = nbformat.read(path, as_version=4)
            client = NotebookClient(nb, timeout=180, kernel_name="course-python", resources={"metadata": {"path": str(path.parent)}})
            client.create_kernel_manager().kernel_spec_manager = manager
            passed = True
            try:
                client.execute()
            except Exception as error:
                passed = False
                (out / f"{path.parent.name}-notebook-error.log").write_text(str(error), encoding="utf-8")
            nbformat.write(nb, out / f"{path.parent.name}.ipynb")
            rows.append({"lesson": path.parent.name, "kind": "notebook", "passed": passed,
                         "seconds": round(time.perf_counter() - start, 2)})
            print(f"{'PASS' if passed else 'FAIL'} notebook {path.parent.name}", flush=True)
    report = {"python": platform.python_version(), "platform": platform.platform(),
              "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "matplotlib")},
              "results": rows}
    (out / f"{args.mode}-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not all(row["passed"] for row in rows):
        raise SystemExit("Some lessons failed; see outputs/verification")


if __name__ == "__main__":
    main()
