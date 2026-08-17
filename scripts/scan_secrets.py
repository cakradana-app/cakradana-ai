"""Refuse to build when a credential is committed.

The failure this prevents is neither subtle nor rare: a key pasted into a config
file, committed, and then present in the history for good. Rotating the key
fixes the access; nothing removes the value from every clone.

Deliberately narrow. A scanner that flags every high-entropy string teaches
people to skip it, at which point it protects nothing. These patterns match
things that are credentials and almost nothing else.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: This file contains the patterns as examples rather than as instances.
EXEMPT = {"scripts/scan_secrets.py"}

BINARY = re.compile(
    r"\.(png|jpe?g|gif|ico|webp|pdf|woff2?|ttf|eot|zip|gz|joblib|pkl|traineddata)$",
    re.IGNORECASE,
)

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "private key block",
        re.compile(r"-----BEGIN (?:RSA|EC|DSA|OPENSSH|PGP) PRIVATE KEY-----"),
    ),
    ("AWS access key id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("OpenAI-style key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("Slack token", re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{10,}\b")),
    (
        "JSON web token",
        re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"),
    ),
    (
        "assigned secret",
        # The placeholder exclusion is what stops this flagging every sample
        # file in the repository, which is what would make it ignorable.
        re.compile(
            r"\b(?:SECRET|TOKEN|PASSWORD|PASSWD|API_KEY|APIKEY|ACCESS_KEY|PRIVATE_KEY)"
            r"\s*[=:]\s*['\"]?"
            r"(?!(?:\$|\{|<|your|xxx|placeholder|example|changeme|test|dummy|fake|\s*$))"
            r"[A-Za-z0-9+/_\-]{16,}",
            re.IGNORECASE,
        ),
    ),
    (
        "connection string with inline password",
        re.compile(
            r"\b(?:mongodb(?:\+srv)?|postgres(?:ql)?|mysql|redis|amqp)://"
            r"[^:\s'\"]+:(?!password\b|changeme\b)[^@\s'\"]{8,}@",
            re.IGNORECASE,
        ),
    ),
]


def tracked_files() -> list[str]:
    output = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [name for name in output.split("\0") if name]


def main() -> int:
    findings: list[tuple[str, int, str, str]] = []

    for name in tracked_files():
        if BINARY.search(name) or name in EXEMPT:
            continue
        try:
            contents = (ROOT / name).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        for number, line in enumerate(contents.splitlines(), start=1):
            for label, pattern in PATTERNS:
                if pattern.search(line):
                    # Never the value itself. A scanner that prints what it
                    # found copies the secret into the build log, which is also
                    # somewhere secrets should not be.
                    findings.append((name, number, label, line.strip()[:24] + "…"))

    if findings:
        print("Possible credentials in tracked files:\n", file=sys.stderr)
        for name, number, label, excerpt in findings:
            print(f"  {name}:{number}  {label}", file=sys.stderr)
            print(f"    {excerpt}", file=sys.stderr)
        print(
            "\nIf any of these is real, rotate it. Removing the line fixes the "
            "file and not the history, and the value is in every clone already.",
            file=sys.stderr,
        )
        return 1

    print("No credentials found in tracked files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
