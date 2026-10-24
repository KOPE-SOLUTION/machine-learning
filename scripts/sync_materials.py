"""Keep runnable README code and notebooks in sync with lesson.py."""
import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START, END = "<!-- CODE:START -->", "<!-- CODE:END -->"



def cell(kind, source, key):
    result = {
        "cell_type": kind, "id": hashlib.sha256(key.encode()).hexdigest()[:12],
        "metadata": {}, "source": source.splitlines(keepends=True),
    }
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    stale = []
    for source in sorted(ROOT.glob("lessons/*/lesson.py")):
        readme = source.with_name("README.md")
        text = readme.read_text(encoding="utf-8")
        before, rest = text.split(START, 1)
        _, after = rest.split(END, 1)
        code = source.read_text(encoding="utf-8").rstrip() + "\n"
        rendered = before + START + "\n```python\n" + code + "```\n" + END + after
        cells = [cell("markdown", before, "intro")]
        chunks = re.split(r"^# %% ?(.*)\n", code, flags=re.MULTILINE)
        if chunks[0].strip():
            cells.append(cell("code", chunks[0], "preamble"))
        for i in range(1, len(chunks), 2):
            cells.append(cell("markdown", "### " + chunks[i] + "\n", f"heading-{i}"))
            cells.append(cell("code", chunks[i + 1], f"code-{i}"))
        cells.append(cell("markdown", after, "outro"))
        nb = {
            "cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5,
        }
        outputs = {readme: rendered, source.with_name("lesson.ipynb"): json.dumps(nb, ensure_ascii=False, indent=1) + "\n"}
        for path, content in outputs.items():
            if args.check:
                if not path.exists() or path.read_text(encoding="utf-8") != content:
                    stale.append(str(path.relative_to(ROOT)))
            else:
                path.write_text(content, encoding="utf-8", newline="\n")
    for source in sorted(ROOT.glob("mojo/*/main.mojo")):
        readme = source.with_name("README.md")
        rendered = readme.read_text(encoding="utf-8")
        for label, language, file in (("MOJO", "mojo", source),
                                      ("PYTHON", "python", source.with_name("reference.py"))):
            start, end = f"<!-- {label}:START -->", f"<!-- {label}:END -->"
            before, rest = rendered.split(start, 1)
            _, after = rest.split(end, 1)
            code = file.read_text(encoding="utf-8").rstrip() + "\n"
            rendered = before + start + f"\n```{language}\n" + code + "```\n" + end + after
        if args.check:
            if readme.read_text(encoding="utf-8") != rendered:
                stale.append(str(readme.relative_to(ROOT)))
        else:
            readme.write_text(rendered, encoding="utf-8", newline="\n")
    if stale:
        raise SystemExit("Out of sync: " + ", ".join(stale))
    print("README code and notebooks:", "in sync" if args.check else "generated")


if __name__ == "__main__":
    main()
