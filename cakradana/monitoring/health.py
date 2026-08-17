"""What the model is doing in production, as a set of numbers.

A model health view is where a failure that produces no errors becomes visible.
The classifier lane that quietly stopped loading, the rule that has been
returning indeterminate for every donation since a register went stale, the
alert volume that has drifted to three times what the team can review — none of
these throw, and none of them appear in a request log.

Everything here is computed from scoring events the service already holds, so it
reports the deployment rather than a training run. Where a figure cannot be
computed from those, it is reported as unavailable with the reason, because a
dashboard that shows zero for something it never measured is worse than one with
a gap in it.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Sequence

from cakradana.scoring.result import Band, Lane, ScoringResult


@dataclass(frozen=True)
class LaneHealth:
    """Whether a lane is running, and what it is contributing when it does."""

    lane: str
    ran: int
    did_not_run: int
    #: Why, when it did not. Kept as counts per distinct reason: "no trained
    #: model is loaded" and "timed out" are different problems and averaging
    #: them into an availability percentage hides which one is happening.
    reasons: dict[str, int] = field(default_factory=dict)
    mean_contribution: float | None = None

    @property
    def total(self) -> int:
        return self.ran + self.did_not_run

    @property
    def availability(self) -> float:
        return self.ran / self.total if self.total else 0.0

    def describe(self) -> str:
        if not self.total:
            return f"{self.lane}: no scoring events"
        if self.ran == 0:
            reason = max(self.reasons, key=self.reasons.get, default="unknown")
            return f"{self.lane}: never ran ({reason})"
        return (
            f"{self.lane}: ran for {self.availability:.0%} of donations, "
            f"mean contribution {self.mean_contribution:.1f}"
        )


@dataclass(frozen=True)
class AlertVolume:
    """How much the system is surfacing against how much can be reviewed."""

    period_days: int
    flagged: int
    budget: int | None

    @property
    def over_budget(self) -> bool:
        return self.budget is not None and self.flagged > self.budget

    def describe(self) -> str:
        if self.budget is None:
            return (
                f"{self.flagged} donation(s) flagged in {self.period_days} days; "
                f"no review budget is configured, so whether that is more than can "
                f"be handled is unknown"
            )
        ratio = self.flagged / self.budget if self.budget else 0
        return (
            f"{self.flagged} flagged against a budget of {self.budget} "
            f"({ratio:.1f}×) over {self.period_days} days"
        )


@dataclass(frozen=True)
class ModelHealth:
    scored: int
    window_days: int
    lanes: tuple[LaneHealth, ...]
    bands: dict[str, int]
    rule_coverage: tuple[tuple[str, int, int], ...]
    alert_volume: AlertVolume
    degraded_share: float
    versions: dict[str, str | None]
    #: Recall is not computable from serving state at all, and saying so is the
    #: honest report. It needs a random sample of unflagged donations reviewed
    #: by people, which is an operational process rather than a metric this
    #: service can derive.
    recall: None = None
    recall_unavailable_reason: str = (
        "recall requires a reviewed random sample of unflagged donations; "
        "it cannot be derived from scoring events, which only describe what "
        "the system surfaced"
    )

    def concerns(self) -> tuple[str, ...]:
        """What a person should look at, in the order they should look at it."""
        found: list[str] = []
        for lane in self.lanes:
            if lane.total and lane.ran == 0:
                found.append(lane.describe())
        for rule_id, evaluated, indeterminate in self.rule_coverage:
            total = evaluated + indeterminate
            if total and indeterminate / total > 0.5:
                found.append(
                    f"{rule_id} could not be evaluated for "
                    f"{indeterminate / total:.0%} of donations"
                )
        if self.alert_volume.over_budget:
            found.append(self.alert_volume.describe())
        if self.degraded_share > 0.25:
            found.append(
                f"{self.degraded_share:.0%} of scores were produced with at least "
                f"one lane unavailable"
            )
        return tuple(found)


def assess(
    events: Sequence[ScoringResult],
    *,
    now: datetime | None = None,
    window_days: int = 30,
    review_budget: int | None = None,
) -> ModelHealth:
    """Summarise recent scoring.

    Bounded to a window because a lane that broke last week is invisible in an
    average taken over a year.
    """
    if events:
        moment = now or max(e.scored_at for e in events)
    else:
        moment = now or datetime.now().astimezone()
    since = moment - timedelta(days=window_days)
    recent = [e for e in events if e.scored_at >= since]

    ran: Counter[str] = Counter()
    absent: Counter[str] = Counter()
    reasons: dict[str, Counter[str]] = defaultdict(Counter)
    contributions: dict[str, list[int]] = defaultdict(list)
    bands: Counter[str] = Counter()
    degraded = 0

    coverage: dict[str, list[int]] = {}

    for event in recent:
        if event.behavioural:
            bands[str(event.behavioural.band)] += 1
            if event.behavioural.degraded:
                degraded += 1
            for lane in event.behavioural.lanes:
                name = str(lane.lane)
                if lane.available:
                    ran[name] += 1
                    contributions[name].append(lane.contribution)
                else:
                    absent[name] += 1
                    reasons[name][lane.unavailable_reason or "unstated"] += 1

        for finding in event.legal_findings:
            entry = coverage.setdefault(finding.rule_id, [0, 0])
            entry[0] += 1
        for indeterminate in event.indeterminate_rules:
            entry = coverage.setdefault(indeterminate.rule_id, [0, 0])
            entry[1] += 1

    lanes = tuple(
        LaneHealth(
            lane=name,
            ran=ran.get(name, 0),
            did_not_run=absent.get(name, 0),
            reasons=dict(reasons.get(name, {})),
            mean_contribution=(
                sum(contributions[name]) / len(contributions[name])
                if contributions.get(name)
                else None
            ),
        )
        for name in sorted({*ran, *absent} or {str(lane) for lane in Lane})
    )

    flagged = sum(
        1
        for event in recent
        if event.legal_findings
        or (event.behavioural and event.behavioural.band in (Band.HIGH, Band.CRITICAL))
    )

    latest = recent[-1] if recent else None
    return ModelHealth(
        scored=len(recent),
        window_days=window_days,
        lanes=lanes,
        bands=dict(bands),
        rule_coverage=tuple(
            (rule_id, evaluated, indeterminate)
            for rule_id, (evaluated, indeterminate) in sorted(coverage.items())
        ),
        alert_volume=AlertVolume(
            period_days=window_days, flagged=flagged, budget=review_budget
        ),
        degraded_share=degraded / len(recent) if recent else 0.0,
        versions={
            "model": latest.versions.model if latest else None,
            "rule_set": latest.versions.rule_set if latest else None,
            "features": latest.versions.features if latest else None,
        },
    )
