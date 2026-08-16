"""Evaluation metrics.

The binding constraint in this domain is analyst hours, not the loss function.
A team can review some number of donations per period, and the only question
that matters is how much genuine risk sits in the ones it actually looks at.
So the metrics here are all defined against a review budget rather than over
the whole population.

Accuracy and F1 are deliberately absent. On a realistic population where a few
percent of donations are risky, a model that flags nothing scores above 95%
accuracy, and reporting that would be worse than reporting nothing.

The metric that decides whether the model ships is lift over the rules. A
classifier trained on heuristic labels can reproduce the heuristics and look
capable while adding nothing, and the only way to see that is to count what it
finds that the rules did not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class Scored:
    """One evaluated donation."""

    donation_id: str
    score: float
    #: Confirmed by a human. Heuristic labels are excluded from evaluation:
    #: measuring a model against the rules it was trained on measures only how
    #: well it memorised them.
    confirmed_risky: bool
    #: Whether any behavioural heuristic flagged this donation. Determines what
    #: counts as an incremental find.
    rule_flagged: bool = False


@dataclass(frozen=True)
class BudgetMetrics:
    """Performance at a fixed review budget."""

    budget: int
    precision_at_b: float
    recall_at_b: float
    lift_at_b: float
    #: Confirmed-risky donations the model surfaced that no rule flagged.
    novel_finds: int
    #: Confirmed-risky donations the rules alone would surface at this budget.
    rule_baseline_finds: int
    total_confirmed_risky: int

    @property
    def model_earns_its_place(self) -> bool:
        """Whether the classifier is worth operating at all.

        A lift at or below parity means the rules alone would surface as much
        as the model adds, and the rules are cheaper, explainable, and already
        built.
        """
        return self.lift_at_b > 1.0

    def describe(self) -> str:
        return (
            f"B={self.budget}: precision={self.precision_at_b:.3f} "
            f"recall={self.recall_at_b:.3f} lift={self.lift_at_b:.2f} "
            f"({self.novel_finds} novel vs {self.rule_baseline_finds} from rules)"
        )


def precision_at_budget(scored: Sequence[Scored], budget: int) -> float:
    top = _top(scored, budget)
    if not top:
        return 0.0
    return sum(1 for s in top if s.confirmed_risky) / len(top)


def recall_at_budget(scored: Sequence[Scored], budget: int) -> float:
    """Share of all confirmed-risky donations that reach the review queue.

    Only meaningful once unflagged donations are sampled and reviewed too.
    Without that sample the denominator counts confirmed cases that were
    confirmed because they were surfaced, which measures the system against its
    own output.
    """
    total = sum(1 for s in scored if s.confirmed_risky)
    if total == 0:
        return 0.0
    return sum(1 for s in _top(scored, budget) if s.confirmed_risky) / total


def lift_at_budget(scored: Sequence[Scored], budget: int) -> BudgetMetrics:
    """Incremental yield over what the heuristics alone would surface."""
    top = _top(scored, budget)
    novel = sum(1 for s in top if s.confirmed_risky and not s.rule_flagged)

    # What the rules would surface on their own with the same budget: the
    # flagged donations, capped at the budget.
    rule_surfaced = [s for s in scored if s.rule_flagged][:budget]
    baseline = sum(1 for s in rule_surfaced if s.confirmed_risky)

    if baseline == 0:
        # With no baseline to beat, any confirmed find is incremental, but a
        # ratio against zero is not a number. Reported as parity when the model
        # also finds nothing, so an empty result never looks like success.
        lift = float(novel) if novel else 0.0
    else:
        lift = novel / baseline

    return BudgetMetrics(
        budget=budget,
        precision_at_b=precision_at_budget(scored, budget),
        recall_at_b=recall_at_budget(scored, budget),
        lift_at_b=lift,
        novel_finds=novel,
        rule_baseline_finds=baseline,
        total_confirmed_risky=sum(1 for s in scored if s.confirmed_risky),
    )


def _top(scored: Sequence[Scored], budget: int) -> list[Scored]:
    return sorted(scored, key=lambda s: s.score, reverse=True)[: max(budget, 0)]


def average_precision(scored: Sequence[Scored]) -> float:
    """Area under the precision-recall curve.

    Reported alongside the budget metrics because it is threshold-free, but it
    is not the decision metric: it describes the whole ranking, and nobody
    reviews the whole ranking.
    """
    ordered = sorted(scored, key=lambda s: s.score, reverse=True)
    total = sum(1 for s in ordered if s.confirmed_risky)
    if total == 0:
        return 0.0
    hits = 0
    accumulated = 0.0
    for index, item in enumerate(ordered, start=1):
        if item.confirmed_risky:
            hits += 1
            accumulated += hits / index
    return accumulated / total


@dataclass(frozen=True)
class CalibrationReport:
    expected_calibration_error: float
    bins: tuple[tuple[float, float, int], ...] = field(default=())

    def describe(self) -> str:
        return f"expected calibration error {self.expected_calibration_error:.4f}"


def calibration_error(
    scored: Sequence[Scored], *, bins: int = 10
) -> CalibrationReport:
    """How far predicted probabilities sit from observed frequencies.

    The claim a score is allowed to make is exactly this: of donations
    historically scored in a band, some measured share were confirmed risky.
    That claim is only true if the mapping has been checked, which is why
    calibration is a requirement rather than a refinement.
    """
    if not scored:
        return CalibrationReport(expected_calibration_error=0.0)

    buckets: list[list[Scored]] = [[] for _ in range(bins)]
    for item in scored:
        index = min(int(item.score * bins), bins - 1)
        buckets[index].append(item)

    total = len(scored)
    error = 0.0
    summary: list[tuple[float, float, int]] = []
    for index, bucket in enumerate(buckets):
        if not bucket:
            continue
        predicted = sum(s.score for s in bucket) / len(bucket)
        observed = sum(1 for s in bucket if s.confirmed_risky) / len(bucket)
        error += (len(bucket) / total) * abs(predicted - observed)
        summary.append((round(predicted, 4), round(observed, 4), len(bucket)))

    return CalibrationReport(
        expected_calibration_error=error, bins=tuple(summary)
    )


def select_threshold(
    scored: Sequence[Scored], *, min_recall_not_risky: float = 0.70
) -> float:
    """Pick an operating threshold that bounds the burden on clean donations.

    Maximises detection of risky donations subject to a floor on how many
    non-risky ones are correctly left alone. The floor is what keeps the
    threshold from drifting to a point that technically catches more while
    burying analysts in donations that turn out to be fine.

    This preserves the selection rule the previous pipeline used, which was
    sound and is the one piece of it worth keeping.
    """
    candidates = sorted({s.score for s in scored})
    if not candidates:
        return 0.5

    risky = [s for s in scored if s.confirmed_risky]
    clean = [s for s in scored if not s.confirmed_risky]
    if not risky or not clean:
        return 0.5

    best_threshold = 1.0
    best_recall = -1.0
    for threshold in candidates:
        recall_clean = sum(1 for s in clean if s.score < threshold) / len(clean)
        if recall_clean < min_recall_not_risky:
            continue
        recall_risky = sum(1 for s in risky if s.score >= threshold) / len(risky)
        if recall_risky > best_recall:
            best_recall = recall_risky
            best_threshold = threshold

    return best_threshold


def analyst_budget(
    *, analysts: int, cases_per_analyst_per_day: int, days: int
) -> int:
    """How many donations a team can actually review in a period.

    Derived from staffing rather than chosen to make a metric look good. If the
    budget is wrong, every precision figure reported against it describes an
    operating point nobody works at.
    """
    if min(analysts, cases_per_analyst_per_day, days) < 0:
        raise ValueError("budget inputs cannot be negative")
    return analysts * cases_per_analyst_per_day * days
