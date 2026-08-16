"""Watching for the ground moving under a deployed model.

A model is fitted to a population. When that population changes — a new
electoral cycle, a new ingestion channel, a scanner replaced — the model keeps
producing confident scores against a world it was not fitted to, and nothing in
its output says so.

Three things are watched, and they fail in different ways.

Input drift says the donations arriving no longer resemble the ones trained on.
Score drift says the model's output distribution has moved, which can happen
without any input drift if a feature quietly changed meaning. Rule coverage
says how much of the statutory picture is actually being evaluated, and it is
the one an operator is most likely to overlook: a register going stale silently
converts a prohibition into a pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from cakradana.features import FeatureVector
from cakradana.rules.engine import RuleEvaluation
from cakradana.schema.enums import RuleOutcome

#: Population stability above this is conventionally read as a material shift.
#: Provisional, and reviewed once enough history exists to know what ordinary
#: month-to-month movement looks like for this data.
DRIFT_THRESHOLD = 0.25


@dataclass(frozen=True)
class FeatureDrift:
    feature: str
    stability_index: float
    #: Share of donations where the feature could not be computed. A jump here
    #: usually means an upstream field stopped arriving, which presents as
    #: drift but is a pipeline fault.
    null_rate: float
    baseline_null_rate: float

    @property
    def has_drifted(self) -> bool:
        return self.stability_index > DRIFT_THRESHOLD

    @property
    def null_rate_jumped(self) -> bool:
        return self.null_rate - self.baseline_null_rate > 0.10

    def describe(self) -> str:
        parts = [f"{self.feature}: stability {self.stability_index:.3f}"]
        if self.null_rate_jumped:
            parts.append(
                f"unavailable in {self.null_rate:.0%} of records, up from "
                f"{self.baseline_null_rate:.0%}"
            )
        return "; ".join(parts)


@dataclass
class DriftReport:
    features: tuple[FeatureDrift, ...] = ()
    score_shift: float | None = None
    findings: list[str] = field(default_factory=list)

    @property
    def drifted_features(self) -> tuple[FeatureDrift, ...]:
        return tuple(f for f in self.features if f.has_drifted)

    @property
    def pipeline_faults(self) -> tuple[FeatureDrift, ...]:
        """Features that stopped being computable, which is not drift.

        Separated because the remedy differs entirely: drift calls for
        retraining, a field that stopped arriving calls for fixing ingestion,
        and retraining on the broken data would bake the fault in.
        """
        return tuple(f for f in self.features if f.null_rate_jumped)

    @property
    def needs_attention(self) -> bool:
        return bool(self.drifted_features or self.pipeline_faults or self.findings)

    def summary(self) -> str:
        if not self.needs_attention:
            return "no material drift detected"
        lines = []
        if self.pipeline_faults:
            lines.append(
                "features that stopped being computable (fix ingestion, do not retrain): "
                + ", ".join(f.feature for f in self.pipeline_faults)
            )
        if self.drifted_features:
            lines.append(
                "features whose distribution moved: "
                + ", ".join(f.describe() for f in self.drifted_features)
            )
        lines.extend(self.findings)
        return "\n".join(lines)


def population_stability(
    baseline: Sequence[float], current: Sequence[float], *, buckets: int = 10
) -> float:
    """How far one distribution has moved from another.

    Quantile buckets are taken from the baseline so the comparison is against
    what the model was fitted to rather than against a moving reference.
    """
    baseline = [v for v in baseline if v is not None and not _is_nan(v)]
    current = [v for v in current if v is not None and not _is_nan(v)]
    if len(baseline) < buckets or not current:
        return 0.0

    ordered = sorted(baseline)
    edges = [
        ordered[min(int(len(ordered) * i / buckets), len(ordered) - 1)]
        for i in range(1, buckets)
    ]

    def distribute(values: Sequence[float]) -> list[float]:
        counts = [0] * buckets
        for value in values:
            index = 0
            while index < len(edges) and value > edges[index]:
                index += 1
            counts[index] += 1
        total = len(values)
        # A vanishing floor keeps an empty bucket from producing an infinite
        # index, which would report a bucket nobody landed in as total drift.
        return [max(count / total, 1e-6) for count in counts]

    expected = distribute(baseline)
    actual = distribute(current)
    return sum(
        (a - e) * math.log(a / e) for e, a in zip(expected, actual)
    )


def _is_nan(value: float) -> bool:
    return isinstance(value, float) and value != value


def compare_features(
    baseline: Sequence[FeatureVector],
    current: Sequence[FeatureVector],
    *,
    features: Iterable[str] | None = None,
) -> tuple[FeatureDrift, ...]:
    if not baseline or not current:
        return ()

    names = list(features) if features else list(baseline[0].values)
    drifts: list[FeatureDrift] = []

    for name in names:
        base_values = [v.values.get(name) for v in baseline]
        curr_values = [v.values.get(name) for v in current]

        base_numeric = [v for v in base_values if isinstance(v, (int, float)) and not isinstance(v, bool)]
        curr_numeric = [v for v in curr_values if isinstance(v, (int, float)) and not isinstance(v, bool)]

        drifts.append(
            FeatureDrift(
                feature=name,
                stability_index=population_stability(base_numeric, curr_numeric),
                null_rate=sum(1 for v in curr_values if v is None) / len(curr_values),
                baseline_null_rate=sum(1 for v in base_values if v is None) / len(base_values),
            )
        )

    return tuple(drifts)


@dataclass(frozen=True)
class RuleCoverage:
    """How much of the statutory picture is actually being evaluated.

    The share of donations a rule could not be evaluated for is a first-class
    operational number. A high rate on a prohibition means the system is not
    checking it at all, and that fact has to be visible rather than hidden
    behind an apparently clean result.
    """

    rule_id: str
    evaluated: int
    indeterminate: int

    @property
    def total(self) -> int:
        return self.evaluated + self.indeterminate

    @property
    def indeterminate_rate(self) -> float:
        return self.indeterminate / self.total if self.total else 0.0

    def describe(self) -> str:
        return (
            f"{self.rule_id}: could not be evaluated for "
            f"{self.indeterminate_rate:.0%} of donations"
        )


def rule_coverage(evaluations: Sequence[RuleEvaluation]) -> tuple[RuleCoverage, ...]:
    counts: dict[str, list[int]] = {}
    for evaluation in evaluations:
        for result in evaluation.results:
            if result.outcome is RuleOutcome.NOT_APPLICABLE:
                continue
            entry = counts.setdefault(result.rule_id, [0, 0])
            if result.outcome is RuleOutcome.INDETERMINATE:
                entry[1] += 1
            else:
                entry[0] += 1

    return tuple(
        RuleCoverage(rule_id=rule_id, evaluated=evaluated, indeterminate=indeterminate)
        for rule_id, (evaluated, indeterminate) in sorted(counts.items())
    )


def detect(
    baseline: Sequence[FeatureVector],
    current: Sequence[FeatureVector],
    *,
    baseline_scores: Sequence[float] = (),
    current_scores: Sequence[float] = (),
    evaluations: Sequence[RuleEvaluation] = (),
    coverage_threshold: float = 0.5,
) -> DriftReport:
    """Compare a recent window against the population a model was fitted to."""
    report = DriftReport(features=compare_features(baseline, current))

    if baseline_scores and current_scores:
        report.score_shift = population_stability(baseline_scores, current_scores)
        if report.score_shift > DRIFT_THRESHOLD:
            report.findings.append(
                f"the distribution of scores has moved (stability "
                f"{report.score_shift:.3f}); the same donations would be ranked "
                f"differently than when this model was fitted"
            )

    for coverage in rule_coverage(evaluations):
        if coverage.indeterminate_rate > coverage_threshold:
            report.findings.append(
                coverage.describe()
                + " — this prohibition is largely unenforced, and its absence "
                "from the findings is not evidence of compliance"
            )

    return report
