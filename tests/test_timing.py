"""Latency and scaling, measured rather than intended.

The wall-clock figures belong to whatever machine runs this, so nothing here
asserts a millisecond budget — a threshold tuned on one developer's laptop fails
on a slower CI runner and gets raised until it stops failing, at which point it
measures nothing.

What is asserted is the scaling property, because it is a fact about the code
and survives the move to other hardware: judging a donation against four times
the history must not cost four times as much. If it does, the system gets slower
as it succeeds, and production is where that is discovered.
"""

from __future__ import annotations

import pytest

from cakradana.data import GeneratorConfig, generate
from cakradana.evaluation.timing import (
    MIN_SAMPLES,
    LatencyReport,
    ScalingReport,
    measure,
    percentile,
)
from cakradana.serving.service import ScoringService


class TestPercentile:
    def test_every_reported_figure_was_actually_observed(self):
        """Interpolation would report a number nothing took. A p95 of 40ms
        should mean some call took 40ms."""
        observed = [1.0, 2.0, 3.0, 4.0, 100.0]
        assert percentile(observed, 0.95) in observed

    def test_the_median_sits_in_the_middle(self):
        assert percentile([1.0, 2.0, 3.0], 0.5) == 2.0

    def test_the_top_percentile_is_the_worst_case(self):
        assert percentile([1.0, 2.0, 3.0, 99.0], 1.0) == 99.0

    def test_an_empty_sample_has_no_percentile(self):
        with pytest.raises(ValueError):
            percentile([], 0.5)


class TestMeasure:
    def test_too_few_samples_is_not_a_measurement(self):
        report = measure("noop", lambda _: None, samples=3, population=10)
        assert report.measured is False
        assert "below the" in (report.unmeasurable_reason or "")

    def test_an_unmeasured_budget_is_unchecked_rather_than_met(self):
        """A budget nothing was measured against has not been met."""
        report = measure("noop", lambda _: None, samples=3, population=10)
        assert report.within(1000.0) is None

    def test_a_measured_run_reports_its_conditions(self):
        report = measure("noop", lambda _: None, samples=MIN_SAMPLES, population=42)
        assert report.measured
        assert report.population == 42
        assert report.machine
        assert str(report.population) in report.describe()

    def test_the_percentiles_are_ordered(self):
        report = measure("noop", lambda i: sum(range(i * 50)), samples=60, population=1)
        assert report.p50_ms <= report.p95_ms <= report.p99_ms <= report.max_ms


class TestScalingArithmetic:
    def report(self, p95: float, population: int) -> LatencyReport:
        return LatencyReport(
            operation="score",
            samples=100,
            population=population,
            p50_ms=p95 / 2,
            p95_ms=p95,
            p99_ms=p95,
            max_ms=p95,
            machine="test",
        )

    def test_cost_growing_more_slowly_than_the_population_is_sublinear(self):
        scaling = ScalingReport(
            "score", self.report(10.0, 500), self.report(15.0, 2000)
        )
        assert scaling.population_ratio == 4.0
        assert scaling.cost_ratio == pytest.approx(1.5)
        assert scaling.is_sublinear is True

    def test_cost_tracking_the_population_is_not_sublinear(self):
        scaling = ScalingReport(
            "score", self.report(10.0, 500), self.report(40.0, 2000)
        )
        assert scaling.is_sublinear is False
        assert "LINEAR OR WORSE" in scaling.describe()

    def test_an_unmeasurable_side_leaves_the_verdict_unknown(self):
        unmeasured = LatencyReport(
            operation="score",
            samples=2,
            population=500,
            p50_ms=0.0,
            p95_ms=0.0,
            p99_ms=0.0,
            max_ms=0.0,
            machine="test",
            unmeasurable_reason="too few samples",
        )
        scaling = ScalingReport("score", unmeasured, self.report(40.0, 2000))
        assert scaling.cost_ratio is None
        assert scaling.is_sublinear is None

    def test_a_baseline_too_small_to_time_yields_no_ratio(self):
        """A ratio against a figure that did not register is not a
        measurement of anything."""
        scaling = ScalingReport(
            "score", self.report(0.0, 500), self.report(40.0, 2000)
        )
        assert scaling.cost_ratio is None


def _service(donations: int):
    dataset = generate(
        GeneratorConfig(
            seed=7,
            n_legitimate_donors=max(20, donations // 6),
            n_recipients=4,
            n_background_donations=donations,
            n_grassroots_campaigns=2,
        )
    )
    service = ScoringService(
        calendar=dataset.calendar,
        registers=dataset.registers,
        entities=dataset.entities,
        require_verified_citations=False,
    )
    service.replay(dataset.donations, entities=dataset.entities)
    return service, dataset.donations


def _timed(donations: int, samples: int = 40):
    """Time the scoring path itself, on a fixed history.

    The scorer is called directly rather than through the HTTP payload, so the
    figure describes rules, features, and lanes rather than JSON parsing. And
    the donation is not remembered, so the history each call is judged against
    stays the size being measured instead of growing under the measurement.
    """
    service, population = _service(donations)

    def call(index: int) -> object:
        donation = population[index % len(population)]
        view = service.store.knowable_at(donation.occurred_at)
        return service.scorer.score(donation, view, entities=service.entities)

    return measure("score", call, samples=samples, population=len(service.store))


@pytest.fixture(scope="module")
def scaling() -> ScalingReport:
    return ScalingReport("score", _timed(400), _timed(1600))


class TestTheRealScoringPath:
    """The measurement that closes the gap: real numbers, on this machine."""

    def test_a_real_latency_figure_exists(self, scaling):
        """The point of this file. The targets were stated and never measured,
        which left them as intent rather than commitment."""
        assert scaling.small.measured
        assert scaling.small.p95_ms > 0
        assert scaling.small.machine

    def test_scoring_is_sublinear_in_the_history_it_is_judged_against(self, scaling):
        """Hardware-independent, so it means the same thing on CI as here. A
        system that costs four times as much on four times the data gets slower
        as it succeeds."""
        assert scaling.is_sublinear is True, scaling.describe()

    def test_the_reading_carries_the_machine_that_produced_it(self, scaling):
        """A latency number without its conditions is not reproducible, and
        gets quoted anyway."""
        assert scaling.small.machine in scaling.small.describe()
        assert str(scaling.small.population) in scaling.small.describe()
