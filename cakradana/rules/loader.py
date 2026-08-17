"""Loading published rule sets.

Rule sets are immutable once published. A change produces a new version rather
than an edit, so that a score recorded last year can still be explained by the
rules that produced it.

That binds everything a verdict depends on: thresholds, tests, applicability,
citations, effective dates, weights, and whether a rule runs at all. It does
not bind ``reason_template``, which is the sentence attached to an outcome and
not part of reaching one — correcting a misleading one changes no verdict this
set has ever issued, and preserving it would only keep the misleading sentence
in front of readers. Amended wording is tracked by the review ledger, which
records the exact sentence each decision was taken on and returns a code to
unreviewed when it changes.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from cakradana.rules.predicates import known_kinds
from cakradana.rules.schema import RuleSet

RULESET_DIR = Path(__file__).parent / "rulesets"


def load_ruleset(path: str | Path) -> RuleSet:
    """Load and validate one rule set file."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    ruleset = RuleSet.model_validate(raw)
    _reject_unknown_tests(ruleset)
    return ruleset


def load_named(version: str) -> RuleSet:
    return load_ruleset(RULESET_DIR / f"{version}.yaml")


def available_versions() -> tuple[str, ...]:
    return tuple(sorted(p.stem for p in RULESET_DIR.glob("*.yaml")))


@lru_cache(maxsize=1)
def load_latest() -> RuleSet:
    """The newest published set.

    Demonstration sets are excluded. They fire rules against fixture reference
    data and produce findings that resemble enforcement without being it, so
    one is only ever loaded by name — never picked up because it happened to
    sort last.
    """
    for version in reversed(available_versions()):
        ruleset = load_named(version)
        if not ruleset.demonstration:
            return ruleset
    raise FileNotFoundError(
        f"no published (non-demonstration) rule set in {RULESET_DIR}"
    )


def _reject_unknown_tests(ruleset: RuleSet) -> None:
    """Fail loading rather than skipping a rule with an unrecognised test.

    A rule whose test kind does not exist would otherwise be loaded and then
    quietly never fire, which presents as a prohibition being enforced when it
    is not.
    """
    kinds = known_kinds()
    unknown = {r.id: r.test.kind for r in ruleset.rules if r.test.kind not in kinds}
    if unknown:
        listed = ", ".join(f"{rid} uses {kind!r}" for rid, kind in sorted(unknown.items()))
        raise ValueError(f"rule set {ruleset.version} references unknown tests: {listed}")
