"""Check local Markdown links and notebook structure without internet access."""
import re
from pathlib import Path
from urllib.parse import unquote

import nbformat

root = Path(__file__).resolve().parents[1]
errors = []
markdown = [root / "README.md", *root.glob("docs/*.md"), *root.glob("lessons/*/README.md"),
            *root.glob("mojo/**/*.md")]
for path in markdown:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    for target in re.findall(r"\]\(([^)]+)\)", text):
        if target.startswith(("https://", "http://", "mailto:", "#")):
            continue
        local = unquote(target.split("#")[0].strip("<>"))
        if local and not (path.parent / local).exists():
            errors.append(f"{path.relative_to(root)} -> {local}")
for path in root.glob("lessons/*/lesson.ipynb"):
    nbformat.validate(nbformat.read(path, as_version=4))
if errors:
    raise SystemExit("Broken local links:\n" + "\n".join(errors))
print(f"Checked {len(markdown)} Markdown files and notebook schemas")
