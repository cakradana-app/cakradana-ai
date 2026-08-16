"""Training and artifact publication.

The pipeline is exercised end to end on a small generated dataset. What is
asserted is not that the model performs well — on generated data that would
mean little — but that the run is honest about what it produced: that the
labels it was measured against are identified, that a model which adds nothing
says so, and that the artifact carries everything needed to reproduce a score.
"""

from __future__ import annotations

import json

import pytest

from cakradana.data import GeneratorConfig, generate
from cakradana.features import FeatureService
from cakradana.history import InMemoryDonationStore
from cakradana.rules import RuleEngine, load_latest
from cakradana.training import (
    HUMAN_LABELS,
    SYNTHETIC_LABELS,
    TrainingConfig,
    build_training_data,
    train,
)
from cakradana.training.registry import ArtifactError, load, save, versions

pytest.importorskip("lightgbm")

TINY = GeneratorConfig(
    seed=99,
    n_legitimate_donors=120,
    n_recipients=4,
    n_background_donations=600,
    n_grassroots_campaigns=2,
)


@pytest.fixture(scope="module")
def context():
    dataset = generate(TINY)
    ruleset = load_latest()
    engine = RuleEngine(
        ruleset,
        calendar=dataset.calendar,
        registers=dataset.registers,
        require_verified_citations=False,
    )
    features = FeatureService(
        ruleset, calendar=dataset.calendar, registers=dataset.registers
    )
    store = InMemoryDonationStore(dataset.donations)
    return dataset, store, engine, features


@pytest.fixture(scope="module")
def result(context):
    dataset, store, engine, features = context
    return train(
        store,
        engine,
        features,
        truth=dataset.truth,
        entities=dataset.entities,
        config=TrainingConfig(review_budget=50, n_estimators=60),
    )


class TestTrainingData:
    def test_class_balance_is_left_alone(self, context):
        """Resampling to look even makes class weighting inert and yields
        precision figures that do not transfer to a rare pattern."""
        dataset, store, engine, features = context
        data = build_training_data(store, engine, features, entities=dataset.entities)
        assert 0.0 < data.base_rate < 0.5
        assert data.scale_pos_weight() > 2.0

    def test_heuristic_positives_carry_reduced_weight(self, context):
        dataset, store, engine, features = context
        data = build_training_data(store, engine, features, entities=dataset.entities)
        positives = [r for r in data.rows if r.label == 1]
        assert positives
        assert all(r.weight <= 0.6 for r in positives)

    def test_rows_record_whether_a_rule_already_flagged_them(self, context):
        dataset, store, engine, features = context
        data = build_training_data(store, engine, features, entities=dataset.entities)
        assert any(r.rule_flagged for r in data.rows)
        assert any(not r.rule_flagged for r in data.rows)


class TestLabelBasis:
    def test_generated_labels_are_not_reportable_as_performance(self, result):
        """Generated labels say whether a model recovered patterns planted for
        it. Reporting that as detection performance is how a demo becomes a
        claim."""
        assert result.label_basis is SYNTHETIC_LABELS
        assert not result.label_basis.reportable_as_system_performance
        assert "not system performance" in result.summary()

    def test_human_labels_are_reportable(self):
        assert HUMAN_LABELS.reportable_as_system_performance


class TestShippingDecision:
    def test_the_run_states_whether_the_model_adds_anything(self, result):
        assert result.should_ship == result.metrics.model_earns_its_place
        assert isinstance(result.metrics.lift_at_b, float)

    def test_a_model_at_or_below_parity_does_not_ship(self, result):
        verdict = "adds incremental detection" if result.should_ship else "do not ship"
        assert verdict in result.summary()


class TestManifest:
    def test_the_manifest_records_what_the_run_depended_on(self, result):
        manifest = result.manifest
        assert manifest["versions"]["features"].startswith("features-")
        assert manifest["versions"]["rule_set"]
        assert manifest["config"]["seed"] == TrainingConfig().seed
        assert manifest["splits"]["train"]["donations"] > 0
        assert "lift_at_b" in manifest["metrics"]

    def test_the_manifest_is_serialisable(self, result):
        assert json.loads(json.dumps(result.manifest))


class TestRegistry:
    def test_an_artifact_round_trips(self, result, context, tmp_path):
        _, _, _, features = context
        save(
            result,
            "test-1",
            feature_names=features.names,
            categorical_features=features.categorical,
            root=tmp_path,
        )
        loaded = load("test-1", root=tmp_path)
        assert loaded.feature_names == features.names
        assert loaded.threshold == pytest.approx(result.threshold)
        assert loaded.feature_set_version == features.version

    def test_a_version_is_never_overwritten(self, result, context, tmp_path):
        """Retraining writes a new version so a score recorded earlier can
        still be reproduced by the artifact that produced it."""
        _, _, _, features = context
        kwargs = dict(
            feature_names=features.names,
            categorical_features=features.categorical,
            root=tmp_path,
        )
        save(result, "test-1", **kwargs)
        with pytest.raises(ArtifactError, match="already exists"):
            save(result, "test-1", **kwargs)

    def test_a_missing_artifact_raises_rather_than_degrading(self, tmp_path):
        """A service that starts without its model and scores anyway fails
        invisibly to whoever reads the output."""
        with pytest.raises(ArtifactError, match="no artifact"):
            load("absent", root=tmp_path)

    def test_versions_lists_only_complete_artifacts(self, result, context, tmp_path):
        _, _, _, features = context
        save(
            result,
            "test-1",
            feature_names=features.names,
            categorical_features=features.categorical,
            root=tmp_path,
        )
        (tmp_path / "half-written").mkdir()
        assert versions(tmp_path) == ("test-1",)


class TestClassifierLane:
    def test_the_lane_refuses_a_vector_missing_required_inputs(
        self, result, context, tmp_path
    ):
        """A null feature is a real state the model was trained on. A feature
        the vector never computed is a defect, and scoring through it
        substitutes a fabricated input."""
        from cakradana.features import FeatureVector
        from cakradana.lanes.classifier import ClassifierLane
        from cakradana.scoring.result import Lane

        dataset, store, engine, features = context
        save(
            result,
            "test-1",
            feature_names=features.names,
            categorical_features=features.categorical,
            root=tmp_path,
        )
        lane = ClassifierLane(load("test-1", root=tmp_path))

        donation = dataset.donations[0]
        truncated = FeatureVector(
            donation_id=donation.donation_id,
            donation_version=1,
            computed_at=donation.occurred_at,
            feature_set_version=features.version,
            values={"amount": 1_000_000},
        )
        outcome = lane.evaluate(None, None, truncated)
        assert outcome.lane is Lane.CLASSIFIER
        assert not outcome.available
        assert "missing" in outcome.unavailable_reason

    def test_the_lane_scores_a_complete_vector(self, result, context, tmp_path):
        from cakradana.lanes.classifier import ClassifierLane

        dataset, store, engine, features = context
        save(
            result,
            "test-1",
            feature_names=features.names,
            categorical_features=features.categorical,
            root=tmp_path,
        )
        lane = ClassifierLane(load("test-1", root=tmp_path))

        donation = dataset.donations[-1]
        view = store.knowable_at(donation.occurred_at)
        vector = features.compute(donation, view, entities=dataset.entities)
        outcome = lane.evaluate(None, None, vector)

        assert outcome.available
        assert 0.0 <= outcome.probability <= 1.0
        assert outcome.reasons, "a score is never surfaced without reasons"
        assert outcome.contribution <= outcome.max_contribution
