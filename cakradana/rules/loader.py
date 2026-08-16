"""Loading published rule sets.

Rule sets are immutable once published. A change produces a new version rather
than an edit, so that a score recorded last year can still be explained by the
rules that produced it.
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
    versions = available_versions()
    if not versions:
        raise FileNotFoundError(f"no rule sets published in {RULESET_DIR}")
    return load_named(versions[-1])


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
