"""Drift detection and the retraining decision.

The case that matters most is the one that looks healthy. Labels accumulate
from reviewed alerts, so a model retrained on them learns to agree with its
previous self: precision on that population rises, every reported number
improves, and coverage of what the model was already missing narrows without
appearing anywhere.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from cakradana.monitoring import (
    MIN_AUDIT_SHARE,
    MIN_HUMAN_LABELS,
    audit_share,
    compare_features,
    detect,
    evaluate,
    population_stability,
    rule_coverage,
)
from cakradana.features import FeatureVector
from cakradana.schema import Label
from cakradana.schema.enums import LabelSource, LabelValue

NOW = datetime.now(timezone.utc)


def label(index: int, *, source=LabelSource.ANALYST_DISPOSITION, note=None) -> Label:
    return Label(
        label_id=f"l{index}",
        donation_id=f"d{index}",
        donation_version=1,
        value=LabelValue.RISKY,
        source=source,
        weight=0.9,
        created_at=NOW,
        note=note,
    )


def vector(index: int, **values) -> FeatureVector:
    return FeatureVector(
        donation_id=f"d{index}",
        donation_version=1,
        computed_at=NOW,
        feature_set_version="features-test",
        values=values,
    )


class TestPopulationStability:
    def test_an_unchanged_population_shows_no_movement(self):
        values = list(range(100))
        assert population_stability(values, values) == pytest.approx(0.0, abs=1e-9)

    def test_a_shifted_population_shows_movement(self):
        assert population_stability(list(range(100)), [v + 80 for v in range(100)]) > 1.0

    def test_too_little_baseline_reports_nothing_rather_than_noise(self):
        assert population_stability([1, 2], list(range(100))) == 0.0

    def test_an_empty_bucket_does_not_report_infinite_drift(self):
        # Without a floor an unoccupied bucket produces an infinite index,
        # reporting a bucket nobody landed in as total drift.
        assert population_stability(list(range(100)), [50] * 100) < float("inf")


class TestFeatureDrift:
    def test_a_stable_feature_is_not_flagged(self):
        baseline = [vector(i, amount=float(i % 50)) for i in range(200)]
        current = [vector(i, amount=float(i % 50)) for i in range(200)]
        drifts = compare_features(baseline, current)
        assert not any(d.has_drifted for d in drifts)

    def test_a_moved_feature_is_flagged(self):
        baseline = [vector(i, amount=float(i % 50)) for i in range(200)]
        current = [vector(i, amount=float(500 + i % 50)) for i in range(200)]
        drifts = compare_features(baseline, current)
        assert any(d.has_drifted for d in drifts)

    def test_a_feature_that_stopped_arriving_is_a_fault_not_drift(self):
        """The remedies differ entirely. Drift calls for retraining; a field
        that stopped arriving calls for fixing ingestion, and retraining on it
        would bake the fault in."""
        baseline = [vector(i, amount=float(i)) for i in range(200)]
        current = [vector(i, amount=None) for i in range(200)]
        report = detect(baseline, current)
        assert report.pipeline_faults
        assert "do not retrain" in report.summary()


class TestRuleCoverage:
    def test_an_unevaluable_rule_is_reported_as_uncovered(self):
        from cakradana.rules.engine import RuleEvaluation, RuleResult
        from cakradana.schema.enums import RuleOutcome

        evaluations = [
            RuleEvaluation(
                donation_id=f"d{i}",
                donation_version=1,
                rule_set_version="rules-test",
                evaluated_at=NOW,
                results=(
                    RuleResult(
                        rule_id="RULE-T1-09",
                        tier=1,
                        outcome=RuleOutcome.INDETERMINATE,
                        reason="register unavailable",
                    ),
                    RuleResult(rule_id="RULE-T1-01", tier=1, outcome=RuleOutcome.PASS),
                ),
            )
            for i in range(10)
        ]
        coverage = {c.rule_id: c for c in rule_coverage(evaluations)}
        assert coverage["RULE-T1-09"].indeterminate_rate == 1.0
        assert coverage["RULE-T1-01"].indeterminate_rate == 0.0

    def test_poor_coverage_is_surfaced_as_an_unenforced_prohibition(self):
        from cakradana.rules.engine import RuleEvaluation, RuleResult
        from cakradana.schema.enums import RuleOutcome

        evaluations = [
            RuleEvaluation(
                donation_id=f"d{i}",
                donation_version=1,
                rule_set_version="rules-test",
                evaluated_at=NOW,
                results=(
                    RuleResult(
                        rule_id="RULE-T1-07",
                        tier=1,
                        outcome=RuleOutcome.INDETERMINATE,
                        reason="jurisdiction unavailable",
                    ),
                ),
            )
            for i in range(10)
        ]
        report = detect([], [], evaluations=evaluations)
        assert any("largely unenforced" in f for f in report.findings)


class TestRetrainingDecision:
    def test_labels_drawn_only_from_reviewed_alerts_block_a_retrain(self):
        """The failure this exists to prevent. A model refitted on the
        donations it already surfaced learns to agree with itself, and every
        reported number improves while its coverage narrows unmeasured."""
        labels = [label(i) for i in range(MIN_HUMAN_LABELS + 50)]
        decision = evaluate(labels, drift_detected=True)
        assert decision.any_trigger_fired
        assert not decision.should_retrain
        assert any("randomly sampled" in b for b in decision.blockers)

    def test_a_random_audit_sample_unblocks_it(self):
        total = MIN_HUMAN_LABELS + 50
        sampled = int(total * (MIN_AUDIT_SHARE + 0.05)) + 1
        labels = [label(i) for i in range(total - sampled)] + [
            label(i, note="audit-sample") for i in range(total - sampled, total)
        ]
        decision = evaluate(labels, drift_detected=True)
        assert decision.should_retrain

    def test_too_few_labels_block_a_retrain(self):
        labels = [label(i, note="audit-sample") for i in range(10)]
        decision = evaluate(labels, drift_detected=True)
        assert not decision.should_retrain
        assert any("mostly reproduces" in b for b in decision.blockers)

    def test_absent_adjudications_are_reported_without_blocking(self):
        """A corpus of analyst judgements is usable on its own, so this is not
        a blocker. But a label set containing no adjudicated outcome was
        assembled without any contested attribution ever being resolved, and
        that reaches nobody if it lives in a comment."""
        labels = [label(i, note="audit-sample") for i in range(MIN_HUMAN_LABELS + 50)]
        decision = evaluate(labels, drift_detected=True)
        assert decision.should_retrain
        assert any("adjudicated dispute outcomes" in n for n in decision.notes)
        assert "worth knowing" in decision.summary()

    def test_a_note_is_not_a_blocker(self):
        labels = [label(i, note="audit-sample") for i in range(MIN_HUMAN_LABELS + 50)]
        decision = evaluate(labels, drift_detected=True)
        assert decision.notes
        assert not decision.blockers

    def test_a_pipeline_fault_blocks_a_retrain(self):
        labels = [label(i, note="audit-sample") for i in range(MIN_HUMAN_LABELS + 50)]
        decision = evaluate(labels, drift_detected=True, pipeline_faults=["amount"])
        assert not decision.should_retrain
        assert any("fit the fault" in b for b in decision.blockers)

    def test_a_settled_system_has_no_reason_to_retrain(self):
        """A recently trained model, no drift, no rule change, and too few new
        judgements to learn from. Retraining here would churn the deployed
        model for nothing."""
        labels = [label(i, note="audit-sample") for i in range(20)]
        decision = evaluate(labels, last_trained_at=NOW)
        assert not decision.any_trigger_fired
        assert not decision.should_retrain

    def test_accumulated_judgements_are_themselves_a_reason_to_retrain(self):
        labels = [
            label(i, note="audit-sample" if i % 4 == 0 else None)
            for i in range(MIN_HUMAN_LABELS + 50)
        ]
        decision = evaluate(labels, last_trained_at=NOW)
        assert decision.should_retrain

    def test_a_rule_change_moves_the_baseline_and_triggers(self):
        """The heuristics produce the training labels, so changing them changes
        what the model is fitted to and what its lift is measured against."""
        labels = [label(i, note="audit-sample") for i in range(MIN_HUMAN_LABELS + 50)]
        decision = evaluate(labels, rule_set_changed=True, last_trained_at=NOW)
        assert decision.should_retrain

    def test_an_ageing_model_triggers(self):
        labels = [label(i, note="audit-sample") for i in range(MIN_HUMAN_LABELS + 50)]
        decision = evaluate(labels, last_trained_at=NOW - timedelta(days=200))
        assert decision.should_retrain

    def test_confirmations_do_not_count_as_human_judgement(self):
        """Confirmation records that a donation occurred, not that it is
        legitimate, so a corpus of them contains no risk verdicts to learn
        from."""
        labels = [
            Label(
                label_id=f"c{i}",
                donation_id=f"d{i}",
                donation_version=1,
                value=LabelValue.INDETERMINATE,
                source=LabelSource.RECIPIENT_CONFIRMATION,
                weight=0.7,
                created_at=NOW,
            )
            for i in range(500)
        ]
        assert audit_share(labels) == 0.0
        decision = evaluate(labels, drift_detected=True)
        assert not decision.should_retrain
