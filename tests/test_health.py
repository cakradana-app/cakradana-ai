"""Model health.

Every failure this reports produces no exception and appears in no request log:
a lane that stopped loading, a rule that has been indeterminate since a register
went stale, an alert volume past what anyone can review.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from cakradana.monitoring.health import assess
from cakradana.scoring.result import (
    Band,
    BehaviouralScore,
    Lane,
    LaneResult,
    Reason,
    ScoringResult,
    Versions,
)
from cakradana.rules.engine import RuleResult
from cakradana.rules.schema import RuleOutcome
from tests.conftest import at


def reason(lane: Lane) -> Reason:
    return Reason(code="X", lane=lane, weight=0.5, statement="something was observed")


def event(
    donation_id: str = "d-1",
    *,
    scored=None,
    lanes=(),
    band: Band = Band.LOW,
    score: int = 10,
    findings=(),
    indeterminate=(),
    degraded: bool = False,
) -> ScoringResult:
    return ScoringResult(
        donation_id=donation_id,
        donation_version=1,
        scored_at=scored or at(2026, 6, 30),
        versions=Versions(model="lgbm-1", rule_set="rules-2026.07", features="f-1"),
        legal_findings=findings,
        indeterminate_rules=indeterminate,
        behavioural=BehaviouralScore(
            score=score,
            band=band,
            lanes=lanes,
            reasons=(reason(Lane.GRAPH),),
            degraded=degraded,
            attainable_max=100,
        ),
    )


def ran(lane: Lane, contribution: int = 10) -> LaneResult:
    return LaneResult(
        lane=lane,
        available=True,
        contribution=contribution,
        max_contribution=30,
        reasons=(reason(lane),),
    )


def absent(lane: Lane, why: str) -> LaneResult:
    return LaneResult(
        lane=lane, available=False, max_contribution=30, unavailable_reason=why
    )


def finding(rule_id: str) -> RuleResult:
    return RuleResult(
        rule_id=rule_id,
        tier=1,
        outcome=RuleOutcome.LEGAL_FINDING,
        explanation="exceeds the limit",
    )


def unevaluated(rule_id: str, why: str = "register unavailable") -> RuleResult:
    return RuleResult(
        rule_id=rule_id,
        tier=1,
        outcome=RuleOutcome.INDETERMINATE,
        explanation=why,
    )


class TestLanes:
    def test_a_lane_that_never_ran_is_a_concern(self):
        """It throws nothing and appears in no log. Without this it is
        invisible until somebody asks why nothing is being flagged."""
        health = assess(
            [
                event(f"d-{i}", lanes=(absent(Lane.CLASSIFIER, "no trained model is loaded"),))
                for i in range(10)
            ]
        )
        assert any("never ran" in c for c in health.concerns())

    def test_the_reasons_are_counted_separately(self):
        """"No trained model" and "timed out" are different problems, and one
        availability percentage hides which is happening."""
        events = [
            event("d-1", lanes=(absent(Lane.CLASSIFIER, "no trained model is loaded"),)),
            event("d-2", lanes=(absent(Lane.CLASSIFIER, "timed out"),)),
            event("d-3", lanes=(absent(Lane.CLASSIFIER, "timed out"),)),
        ]
        classifier = next(
            l for l in assess(events).lanes if l.lane == str(Lane.CLASSIFIER)
        )
        assert classifier.reasons == {"no trained model is loaded": 1, "timed out": 2}

    def test_a_lane_that_runs_reports_what_it_contributes(self):
        health = assess([event(f"d-{i}", lanes=(ran(Lane.GRAPH, 12),)) for i in range(5)])
        graph = next(l for l in health.lanes if l.lane == str(Lane.GRAPH))
        assert graph.availability == 1.0
        assert graph.mean_contribution == 12

    def test_a_partly_available_lane_is_not_a_concern_on_its_own(self):
        events = [event("d-1", lanes=(ran(Lane.GRAPH),))] + [
            event("d-2", lanes=(absent(Lane.GRAPH, "timed out"),))
        ]
        assert not any("never ran" in c for c in assess(events).concerns())


class TestRuleCoverage:
    def test_a_rule_that_cannot_be_evaluated_surfaces(self):
        """A high indeterminate rate on a prohibition means the system is not
        checking it, and an empty findings list looks identical to compliance."""
        events = [
            event(f"d-{i}", indeterminate=(unevaluated("RULE-T1-09"),)) for i in range(10)
        ]
        assert any("RULE-T1-09" in c for c in assess(events).concerns())

    def test_a_rule_that_mostly_evaluates_does_not(self):
        events = [event(f"d-{i}", findings=(finding("RULE-T1-01"),)) for i in range(9)]
        events.append(event("d-10", indeterminate=(unevaluated("RULE-T1-01"),)))
        assert not any("RULE-T1-01" in c for c in assess(events).concerns())

    def test_coverage_counts_both_outcomes(self):
        events = [
            event("d-1", findings=(finding("RULE-T1-01"),)),
            event("d-2", indeterminate=(unevaluated("RULE-T1-01"),)),
        ]
        coverage = dict((r, (e, i)) for r, e, i in assess(events).rule_coverage)
        assert coverage["RULE-T1-01"] == (1, 1)


class TestAlertVolume:
    def test_volume_beyond_the_budget_is_a_concern(self):
        """A queue longer than a team can process has a tail nobody reads, and
        every precision figure quoted against it describes an operating point
        that does not exist."""
        events = [
            event(f"d-{i}", band=Band.CRITICAL, score=90, lanes=(ran(Lane.GRAPH),))
            for i in range(60)
        ]
        health = assess(events, review_budget=50)
        assert health.alert_volume.over_budget
        assert any("budget" in c for c in health.concerns())

    def test_without_a_budget_it_says_the_question_is_open(self):
        # Rather than reporting no problem, which is a different claim.
        health = assess([event("d-1", band=Band.CRITICAL, score=90)])
        assert not health.alert_volume.over_budget
        assert "unknown" in health.alert_volume.describe()


class TestWindow:
    def test_older_events_fall_outside_the_window(self):
        """A lane that broke last week is invisible in an average taken over a
        year."""
        old = [
            event(f"old-{i}", scored=at(2025, 1, 1), lanes=(ran(Lane.GRAPH),))
            for i in range(50)
        ]
        recent = [
            event(f"new-{i}", scored=at(2026, 6, 30), lanes=(absent(Lane.GRAPH, "timed out"),))
            for i in range(5)
        ]
        health = assess(old + recent, window_days=30)
        assert health.scored == 5
        graph = next(l for l in health.lanes if l.lane == str(Lane.GRAPH))
        assert graph.ran == 0


class TestRecall:
    def test_recall_is_reported_as_unavailable_with_its_reason(self):
        """It cannot be derived from scoring events, which describe only what
        the system surfaced. A dashboard showing a number here would be
        describing the system's opinion of itself."""
        health = assess([event("d-1")])
        assert health.recall is None
        assert "random sample" in health.recall_unavailable_reason


class TestDegradation:
    def test_widespread_degradation_is_a_concern(self):
        events = [
            event(f"d-{i}", degraded=True, lanes=(absent(Lane.CLASSIFIER, "timed out"),))
            for i in range(10)
        ]
        assert any("unavailable" in c for c in assess(events).concerns())

    def test_a_clean_deployment_has_nothing_to_report(self):
        events = [event(f"d-{i}", lanes=(ran(Lane.GRAPH),)) for i in range(10)]
        assert assess(events, review_budget=50).concerns() == ()
