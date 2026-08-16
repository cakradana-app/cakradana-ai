"""Splits and metrics.

These assert the properties that make a reported number trustworthy: that no
donor crosses a split boundary, that the metrics describe a review budget
rather than the whole population, and that a model which merely reproduces the
rules is visibly worth nothing.
"""

from __future__ import annotations

import pytest

from cakradana.evaluation import (
    LeakageError,
    Scored,
    analyst_budget,
    assert_no_leakage,
    average_precision,
    calibration_error,
    donor_cohort_split,
    lift_at_budget,
    precision_at_budget,
    recall_at_budget,
    select_threshold,
)
from cakradana.evaluation.splits import Split, SplitSet
from tests.conftest import at, make_donation


class TestSplits:
    def test_no_donor_appears_in_two_splits(self):
        donations = [
            make_donation(
                donation_id=f"d{i}",
                sender=f"donor-{i % 40}",
                occurred=at(2026, 1, 1 + (i % 28)),
            )
            for i in range(400)
        ]
        splits = donor_cohort_split(donations)
        assert not splits.train.donors & splits.test.donors
        assert not splits.train.donors & splits.calibration.donors
        assert not splits.calibration.donors & splits.test.donors

    def test_every_donation_is_kept(self):
        donations = [
            make_donation(donation_id=f"d{i}", sender=f"donor-{i % 40}")
            for i in range(200)
        ]
        splits = donor_cohort_split(donations)
        total = len(splits.train) + len(splits.calibration) + len(splits.test)
        assert total == len(donations)

    def test_leakage_is_raised_not_reported(self):
        """A leakage rate that is merely printed gets read, noted, and lived
        with, and the resulting metric looks exactly like an honest one."""
        shared = make_donation(donation_id="d1", sender="donor-1")
        splits = SplitSet(
            train=Split("train", (shared,)),
            calibration=Split("calibration", ()),
            test=Split("test", (shared,)),
        )
        with pytest.raises(LeakageError, match="both train and test"):
            assert_no_leakage(splits)

    def test_a_dedicated_calibration_split_exists(self):
        """Calibrating on training data reproduces the model's own
        overconfidence; calibrating on test data spends the only honest
        performance estimate there is."""
        donations = [
            make_donation(donation_id=f"d{i}", sender=f"donor-{i}")
            for i in range(100)
        ]
        splits = donor_cohort_split(donations)
        assert len(splits.calibration) > 0
        assert not splits.calibration.donors & splits.train.donors

    def test_empty_input_is_rejected(self):
        with pytest.raises(ValueError):
            donor_cohort_split([])


class TestBudgetMetrics:
    def test_precision_counts_only_what_gets_reviewed(self):
        scored = [
            Scored(f"d{i}", score=1.0 - i / 100, confirmed_risky=i < 3)
            for i in range(100)
        ]
        assert precision_at_budget(scored, 10) == pytest.approx(0.3)

    def test_recall_is_against_all_confirmed_risky(self):
        scored = [
            Scored(f"d{i}", score=1.0 - i / 100, confirmed_risky=i < 10)
            for i in range(100)
        ]
        assert recall_at_budget(scored, 5) == pytest.approx(0.5)

    def test_a_budget_larger_than_the_data_is_harmless(self):
        scored = [Scored("d1", score=0.9, confirmed_risky=True)]
        assert precision_at_budget(scored, 500) == 1.0


class TestLift:
    def test_a_model_that_only_reproduces_the_rules_earns_nothing(self):
        """Training on heuristic labels can produce a model that looks capable
        while adding nothing, and only counting incremental finds shows it."""
        scored = [
            Scored(f"d{i}", score=1.0 - i / 20, confirmed_risky=i < 5, rule_flagged=i < 5)
            for i in range(20)
        ]
        metrics = lift_at_budget(scored, 10)
        assert metrics.novel_finds == 0
        assert not metrics.model_earns_its_place

    def test_incremental_finds_produce_lift_above_parity(self):
        scored = [
            # Three the rules already catch, plus four they miss entirely.
            Scored("r1", 0.99, True, rule_flagged=True),
            Scored("r2", 0.98, True, rule_flagged=True),
            Scored("r3", 0.97, True, rule_flagged=True),
            Scored("n1", 0.96, True, rule_flagged=False),
            Scored("n2", 0.95, True, rule_flagged=False),
            Scored("n3", 0.94, True, rule_flagged=False),
            Scored("n4", 0.93, True, rule_flagged=False),
            Scored("c1", 0.10, False, rule_flagged=False),
        ]
        metrics = lift_at_budget(scored, 8)
        assert metrics.novel_finds == 4
        assert metrics.rule_baseline_finds == 3
        assert metrics.model_earns_its_place

    def test_finding_nothing_never_reads_as_success(self):
        scored = [Scored(f"d{i}", 0.5, False) for i in range(10)]
        assert lift_at_budget(scored, 5).lift_at_b == 0.0


class TestCalibration:
    def test_a_perfectly_calibrated_set_has_no_error(self):
        scored = [Scored(f"a{i}", 0.0, False) for i in range(50)] + [
            Scored(f"b{i}", 1.0, True) for i in range(50)
        ]
        assert calibration_error(scored).expected_calibration_error == pytest.approx(0.0)

    def test_overconfidence_shows_up_as_error(self):
        scored = [Scored(f"d{i}", 0.95, False) for i in range(100)]
        assert calibration_error(scored).expected_calibration_error > 0.9


class TestThresholdSelection:
    def test_the_threshold_respects_the_floor_on_clean_donations(self):
        """Without the floor a threshold drifts to a point that catches more
        while burying analysts in donations that turn out to be fine."""
        scored = [Scored(f"r{i}", 0.9, True) for i in range(10)] + [
            Scored(f"c{i}", 0.1 + i / 200, False) for i in range(100)
        ]
        threshold = select_threshold(scored, min_recall_not_risky=0.70)
        left_alone = sum(1 for s in scored if not s.confirmed_risky and s.score < threshold)
        assert left_alone / 100 >= 0.70

    def test_a_degenerate_set_returns_a_neutral_threshold(self):
        assert select_threshold([Scored("d1", 0.4, False)]) == 0.5


class TestAveragePrecision:
    def test_a_perfect_ranking_scores_one(self):
        scored = [Scored(f"d{i}", 1.0 - i / 10, confirmed_risky=i < 3) for i in range(10)]
        assert average_precision(scored) == pytest.approx(1.0)

    def test_no_positives_scores_zero(self):
        assert average_precision([Scored("d1", 0.5, False)]) == 0.0


class TestBudgetDerivation:
    def test_the_budget_comes_from_staffing(self):
        """A budget chosen to make a metric look good describes an operating
        point nobody works at."""
        assert analyst_budget(analysts=3, cases_per_analyst_per_day=10, days=20) == 600

    def test_negative_inputs_are_rejected(self):
        with pytest.raises(ValueError):
            analyst_budget(analysts=-1, cases_per_analyst_per_day=10, days=20)
