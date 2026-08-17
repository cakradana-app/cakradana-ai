#!/usr/bin/env python3
"""Every runtime dependency is pinned, and pinned inside its declared range.

Two files describe what this package needs, and they describe it differently.
``pyproject.toml`` states compatibility — the range of versions the code is
written against — and is what a consumer installing the library resolves
against. ``requirements.txt`` states the exact set a deployment installs.

The failure this catches is a dependency present in one and absent from the
other. ``numpy`` was declared in ``pyproject.toml`` and missing from
``requirements.txt``, so every deployment resolved whatever numpy pip happened
to pick that day, under a package whose stated reason for pinning is that a
scoring result must be reproducible.

It also catches a pin that has drifted outside its own declared range, which is
the same defect read from the other end: the code says it needs one thing and
the deployment installs another.

Parses rather than imports, so it runs before the install it is checking.
``tomllib`` is 3.11+ and this package supports 3.10, so pyproject is read with a
regex over the dependency arrays rather than a TOML parser.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Distributions a deployment does not install, so a missing pin is not a
#: defect. Kept explicit rather than inferred: an inferred exemption grows.
NOT_DEPLOYED: frozenset[str] = frozenset()


def normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def declared(pyproject: str, section: str) -> dict[str, str]:
    """Requirement strings from one dependency array, by distribution name."""
    match = re.search(rf"^{section}\s*=\s*\[(.*?)\]", pyproject, re.S | re.M)
    if not match:
        return {}
    found: dict[str, str] = {}
    for line in re.findall(r'"([^"]+)"', match.group(1)):
        name = re.split(r"[<>=!~\[; ]", line, maxsplit=1)[0]
        found[normalise(name)] = line
    return found


def pinned(requirements: str) -> dict[str, str]:
    """Exact pins from a requirements file, by distribution name."""
    found: dict[str, str] = {}
    for raw in requirements.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        if "==" not in line:
            found[normalise(re.split(r"[<>=!~\[; ]", line, maxsplit=1)[0])] = ""
            continue
        name, version = line.split("==", 1)
        found[normalise(name)] = version.strip()
    return found


def version_tuple(version: str) -> tuple[int, ...]:
    """Numeric components only.

    Enough to compare release versions against the bounds this project writes,
    which are all plain ``major.minor``. A pre-release or a local version would
    compare as its release, which is why one is refused outright below.
    """
    return tuple(int(part) for part in re.findall(r"\d+", version))


def satisfies(version: str, requirement: str) -> str | None:
    """Whether a pinned version sits inside a declared range."""
    if re.search(r"[a-zA-Z]", version.split("+")[0].lstrip("0123456789.")):
        return f"{version} is not a plain release version"
    actual = version_tuple(version)
    for operator, bound in re.findall(r"(>=|<=|==|<|>|!=)\s*([\w.]+)", requirement):
        expected = version_tuple(bound)
        width = max(len(actual), len(expected))
        left = actual + (0,) * (width - len(actual))
        right = expected + (0,) * (width - len(expected))
        ok = {
            ">=": left >= right,
            "<=": left <= right,
            "==": left == right,
            "<": left < right,
            ">": left > right,
            "!=": left != right,
        }[operator]
        if not ok:
            return f"{version} does not satisfy {operator}{bound}"
    return None


def check(section: str, requirements_file: str) -> list[str]:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    requirements = (ROOT / requirements_file).read_text(encoding="utf-8")

    wanted = declared(pyproject, section)
    have = pinned(requirements)
    problems: list[str] = []

    for name, requirement in sorted(wanted.items()):
        if name in NOT_DEPLOYED:
            continue
        if name not in have:
            problems.append(
                f"{name} is declared in pyproject.toml but not pinned in "
                f"{requirements_file}; deployments would resolve it freshly"
            )
            continue
        if not have[name]:
            problems.append(
                f"{name} appears in {requirements_file} without an == pin"
            )
            continue
        conflict = satisfies(have[name], requirement)
        if conflict:
            problems.append(
                f"{name}: {requirements_file} pins {have[name]} but "
                f"pyproject.toml declares {requirement!r} — {conflict}"
            )

    for name in sorted(set(have) - set(wanted)):
        problems.append(
            f"{name} is pinned in {requirements_file} but not declared in "
            f"pyproject.toml; nothing states why it is installed"
        )
    return problems


def main() -> int:
    problems = check("dependencies", "requirements.txt")
    # The dev array is checked against the file that includes the runtime one,
    # so the runtime pins appear in both; only the dev additions are compared.
    dev_pyproject = declared(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8"), "dev"
    )
    dev_pinned = pinned((ROOT / "requirements-dev.txt").read_text(encoding="utf-8"))
    for name, requirement in sorted(dev_pyproject.items()):
        if name not in dev_pinned:
            problems.append(
                f"{name} is declared as a dev dependency but not pinned in "
                f"requirements-dev.txt"
            )
        elif conflict := satisfies(dev_pinned[name], requirement):
            problems.append(f"{name}: {conflict}")

    if problems:
        print("Dependency pinning problems:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    total = len(declared((ROOT / "pyproject.toml").read_text(encoding="utf-8"), "dependencies"))
    print(f"All {total} runtime dependencies pinned within their declared ranges.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
