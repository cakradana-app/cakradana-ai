"""The anomaly lane."""

from __future__ import annotations

import pytest

pytest.importorskip("sklearn")

from cakradana.data import GeneratorConfig, generate  # noqa: E402
from cakradana.features import FeatureService  # noqa: E402
from cakradana.history import InMemoryDonationStore  # noqa: E402
from cakradana.lanes.anomaly import AnomalyLane, fit  # noqa: E402
from cakradana.rules import RuleEngine, load_latest  # noqa: E402
from cakradana.scoring.result import Lane  # noqa: E402

TINY = GeneratorConfig(
    seed=7,
    n_legitimate_donors=120,
    n_recipients=4,
    n_background_donations=600,
    n_grassroots_campaigns=2,
)


@pytest.fixture(scope="module")
def fitted():
    dataset = generate(TINY)
    ruleset = load_latest()
    features = FeatureService(
        ruleset, calendar=dataset.calendar, registers=dataset.registers
    )
    engine = RuleEngine(
        ruleset,
        calendar=dataset.calendar,
        registers=dataset.registers,
        require_verified_citations=False,
    )
    store = InMemoryDonationStore(dataset.donations)
    vectors = [
        features.compute(d, store.knowable_at(d.occurred_at), entities=dataset.entities)
        for d in dataset.donations
    ]
    return dataset, store, engine, features, vectors, fit(vectors, features)


class TestFitting:
    def test_a_handful_of_records_is_refused(self, fitted):
        """An outlier detector fitted on a handful of records describes the
        handful, not what is ordinary."""
        _, _, _, features, vectors, _ = fitted
        with pytest.raises(ValueError, match="too few donations"):
            fit(vectors[:5], features)

    def test_fill_values_are_fixed_at_fit_time(self, fitted):
        """Recomputing stand-ins at scoring time would substitute different
        values than fitting did, and scoring one donation would derive a
        median from that donation alone."""
        _, _, _, features, _, model = fitted
        assert len(model.fill_values) == len(model.feature_names)
        assert model.feature_names == tuple(features.numeric)

    def test_the_cutoff_comes_from_the_fitted_population(self, fitted):
        _, _, _, _, _, model = fitted
        assert model.cutoff > 0


class TestLaneBehaviour:
    def test_most_donations_are_not_surfaced(self, fitted):
        """Unusual is not suspicious. A lane that flags a large share of
        ordinary traffic spends analyst trust it cannot get back."""
        dataset, store, engine, features, vectors, model = fitted
        lane = AnomalyLane(model)
        surfaced = 0
        for donation, vector in list(zip(dataset.donations, vectors))[:300]:
            view = store.knowable_at(donation.occurred_at)
            evaluation = engine.evaluate(donation, view, entities=dataset.entities)
            outcome = lane.evaluate(evaluation, None, vector)
            if outcome.available and outcome.contribution > 0:
                surfaced += 1
        assert surfaced < 60

    def test_the_lane_is_capped_below_the_classifier(self, fitted):
        dataset, store, engine, features, vectors, model = fitted
        lane = AnomalyLane(model)
        outcome = lane.evaluate(None, None, vectors[0])
        assert outcome.max_contribution == 15

    def test_a_donation_with_a_legal_finding_is_skipped(self, fitted):
        """Reporting that a statutory breach is also statistically unusual
        adds nothing an analyst can act on, and spends a capped budget."""
        dataset, store, engine, features, vectors, model = fitted
        lane = AnomalyLane(model)

        for donation, vector in zip(dataset.donations, vectors):
            view = store.knowable_at(donation.occurred_at)
            evaluation = engine.evaluate(donation, view, entities=dataset.entities)
            if evaluation.legal_findings:
                outcome = lane.evaluate(evaluation, None, vector)
                assert not outcome.available
                assert "legal finding" in outcome.unavailable_reason
                return
        pytest.skip("no legal findings in this dataset to exercise the skip")

    def test_a_surfaced_donation_carries_a_reason(self, fitted):
        dataset, store, engine, features, vectors, model = fitted
        lane = AnomalyLane(model)
        for vector in vectors:
            outcome = lane.evaluate(None, None, vector)
            if outcome.contribution > 0:
                assert outcome.reasons
                assert outcome.reasons[0].lane is Lane.ANOMALY
                assert outcome.reasons[0].comparison
                return
        pytest.fail("the lane surfaced nothing at all")
