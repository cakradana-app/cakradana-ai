"""Assert that every package the source imports is declared.

An undeclared import produces a module that works on the machine it was written
on and fails on a clean install. That is not a build problem — it is a
reproducibility problem, and it is how a documented setup comes to be one that
nobody can follow. This project already had it: the generator imported `faker`,
which appeared in no manifest, so the documented command failed on any fresh
environment.

Checked by parsing rather than by importing, so the check itself needs nothing
installed.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "cakradana"
TESTS = ROOT / "tests"

#: Import names that differ from their distribution name.
DISTRIBUTION_OF = {
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
    "dateutil": "python-dateutil",
}


def top_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            # Relative imports have no module to declare.
            if node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
    return found


def declared() -> set[str]:
    """Every requirement string in pyproject, whichever list it sits in.

    Read with a regular expression rather than a TOML parser: `tomllib` arrived
    in 3.11 and this project supports 3.10, so a parser-based check would fail
    on the oldest version it claims to run on — which is exactly the class of
    problem this script exists to catch.
    """
    text = (ROOT / "pyproject.toml").read_text()
    names: set[str] = set()
    # Requirement strings look like "package>=1.2,<2" and only ever appear
    # inside dependency arrays in this file.
    for entry in re.findall(r'"([A-Za-z0-9_.\-]+(?:[<>=!~][^"]*)?)"', text):
        name = entry.split(";")[0]
        for separator in (">=", "<=", "==", "~=", ">", "<", "!="):
            name = name.split(separator)[0]
        names.add(name.strip().lower().replace("_", "-"))
    return names


def stdlib() -> set[str]:
    return set(sys.stdlib_module_names)


def main() -> int:
    declared_names = declared()
    known = stdlib() | {"cakradana", "tests"}

    missing: dict[str, set[str]] = {}
    for path in [*PACKAGE.rglob("*.py"), *TESTS.rglob("*.py")]:
        for module in top_level_imports(path):
            if module in known:
                continue
            distribution = DISTRIBUTION_OF.get(module, module).lower().replace("_", "-")
            if distribution not in declared_names:
                missing.setdefault(distribution, set()).add(
                    str(path.relative_to(ROOT))
                )

    if missing:
        print("Imported but not declared in pyproject.toml:", file=sys.stderr)
        for distribution, files in sorted(missing.items()):
            print(f"  {distribution}", file=sys.stderr)
            for file in sorted(files):
                print(f"    {file}", file=sys.stderr)
        print(
            "\nAn undeclared import works where it was written and fails on a "
            "clean install, which is how a documented setup becomes one nobody "
            "can follow.",
            file=sys.stderr,
        )
        return 1

    print(f"All imports declared ({len(declared_names)} distributions).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
