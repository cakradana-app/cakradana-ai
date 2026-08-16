"""The two-tier rule engine.

Tier 1 evaluates statutory compliance and produces legal findings with
citations. Tier 2 evaluates behavioural heuristics and produces training labels
and ranking signals.

Keeping them apart is what lets the classifier add anything. Trained on Tier-1
outcomes it could only relearn arithmetic it was already given, and applied to
donations Tier 1 had cleared it would return negatives by construction. Trained
on Tier-2 heuristics it has a target the statute does not already determine.
"""

from cakradana.rules.context import Limit, LimitTable, PeriodWindow, RuleContext
from cakradana.rules.engine import RuleEngine, RuleEvaluation, RuleResult
from cakradana.rules.loader import (
    available_versions,
    load_latest,
    load_named,
    load_ruleset,
)
from cakradana.rules.schema import Rule, RuleSet

__all__ = [
    "Limit",
    "LimitTable",
    "PeriodWindow",
    "Rule",
    "RuleContext",
    "RuleEngine",
    "RuleEvaluation",
    "RuleResult",
    "RuleSet",
    "available_versions",
    "load_latest",
    "load_named",
    "load_ruleset",
]
