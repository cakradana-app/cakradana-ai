#!/usr/bin/env python3
"""The resolved lock agrees with the manifest, and every entry carries a hash.

``check_pins.py`` establishes that the direct dependencies are pinned and that
their pins sit inside the ranges the package declares. That leaves the larger
half of an install unstated: the transitive set. ``fastapi==0.111.0`` is exact,
but it pulls starlette, anyio, and thirty more, and none of those were fixed by
anything — so two installs of the same commit, a month apart, did not
necessarily produce the same code. For a package whose stated reason for
pinning is that a scoring result must be reproducible, that is the part that
matters most, because the arithmetic lives downstream of every one of them.

``requirements.lock`` and ``requirements-dev.lock`` are the fully resolved sets,
each entry pinned exactly and carrying the hashes of the artifacts that satisfy
it. Installing with ``--require-hashes`` then fails on anything the index serves
that is not byte-for-byte what was resolved here — which covers an index
compromise, a re-uploaded artifact, and a mirror serving something else, none of
which a version pin alone notices.

What this script checks is the property that makes the lock worth having, and
which nothing else would notice going wrong:

  - Every pin in the manifest appears in the lock at that exact version. A lock
    regenerated before a manifest edit, or not regenerated after one, installs a
    version nobody chose while both files look maintained.
  - Every requirement in the lock carries at least one hash. A single entry
    without one disables ``--require-hashes`` for the entire install, so the
    guarantee is all-or-nothing and the failure is silent.
  - The dev lock covers the runtime lock, since ``requirements-dev.txt``
    includes ``requirements.txt``: a test run and a deployment resolving
    different versions of the same library is how a suite passes against code
    that is not what ships.

It parses rather than installs, so it runs in seconds and needs no network.
Regenerating after a manifest change is::

    python -m piptools compile --generate-hashes --strip-extras \\
        --output-file requirements.lock requirements.txt
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PAIRS = [
    (ROOT / "requirements.txt", ROOT / "requirements.lock"),
    (ROOT / "requirements-dev.txt", ROOT / "requirements-dev.lock"),
]


def normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def manifest_pins(path: Path) -> dict[str, str]:
    """The exact pins a manifest states, ignoring its includes and comments."""
    pins: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        match = re.match(r"^([A-Za-z0-9._-]+)\s*==\s*([^\s;]+)", line)
        if match:
            pins[normalise(match.group(1))] = match.group(2)
    return pins


def lock_entries(path: Path) -> dict[str, tuple[str, int]]:
    """Each locked distribution, its version, and how many hashes it carries.

    A lock is line-continued: the requirement is on one line and its hashes on
    the ones after it, each ending in a backslash except the last. Tracking the
    current requirement across those continuations is what makes a hash
    countable against the thing it belongs to.
    """
    entries: dict[str, tuple[str, int]] = {}
    current: str | None = None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("--hash="):
            if current is not None:
                version, count = entries[current]
                entries[current] = (version, count + 1)
            continue
        match = re.match(r"^([A-Za-z0-9._-]+)\s*==\s*([^\s\\;]+)", line)
        if match:
            current = normalise(match.group(1))
            entries[current] = (match.group(2), 0)
            # A hash can sit on the same line as the requirement.
            entries[current] = (match.group(2), line.count("--hash="))
        elif not line.startswith("-"):
            current = None
    return entries


def main() -> int:
    problems: list[str] = []

    for manifest, lock in PAIRS:
        if not lock.exists():
            problems.append(
                f"{lock.name} does not exist. Without it the transitive set is "
                f"resolved freshly on every install, which is what pinning "
                f"{manifest.name} exists to prevent."
            )
            continue

        pins = manifest_pins(manifest)
        entries = lock_entries(lock)

        for name, version in sorted(pins.items()):
            if name not in entries:
                problems.append(
                    f"{lock.name}: {name} is pinned in {manifest.name} and absent "
                    f"from the lock. Regenerate the lock."
                )
            elif entries[name][0] != version:
                problems.append(
                    f"{lock.name}: {name} is pinned at {version} in "
                    f"{manifest.name} and locked at {entries[name][0]}. The "
                    f"deployment installs the locked version, so the manifest "
                    f"is describing something that is not installed."
                )

        # One entry without a hash turns the whole install back into an
        # unverified one, because pip applies --require-hashes to the file as a
        # whole. The failure is a guarantee quietly not held, so it is named per
        # entry rather than counted.
        for name, (version, hashes) in sorted(entries.items()):
            if hashes == 0:
                problems.append(
                    f"{lock.name}: {name}=={version} carries no hash. A single "
                    f"unhashed entry disables --require-hashes for every other "
                    f"entry in the file."
                )

        if not entries:
            problems.append(f"{lock.name} resolved to nothing; it is not a lock.")

    runtime = lock_entries(ROOT / "requirements.lock") if (ROOT / "requirements.lock").exists() else {}
    dev = lock_entries(ROOT / "requirements-dev.lock") if (ROOT / "requirements-dev.lock").exists() else {}
    for name, (version, _) in sorted(runtime.items()):
        if name not in dev:
            problems.append(
                f"requirements-dev.lock is missing {name}, which the runtime lock "
                f"holds. The suite would run without a library the deployment has."
            )
        elif dev[name][0] != version:
            problems.append(
                f"{name} is locked at {version} for the deployment and "
                f"{dev[name][0]} for the tests. The suite passes against code "
                f"that is not what ships."
            )

    if problems:
        print("Lock files disagree with the manifests:\n", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print(
            "\nRegenerate with:\n"
            "  python -m piptools compile --generate-hashes --strip-extras "
            "--output-file requirements.lock requirements.txt\n"
            "  python -m piptools compile --generate-hashes --strip-extras "
            "--output-file requirements-dev.lock requirements-dev.txt",
            file=sys.stderr,
        )
        return 1

    total = len(lock_entries(ROOT / "requirements-dev.lock"))
    print(
        f"Lock files agree with the manifests: {total} distributions pinned with "
        f"hashes."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
