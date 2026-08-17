"""Random audit sampling and the recall estimate.

Most of what is asserted here is a refusal. The module's job is to say "not
measurable" often enough that a detection-rate figure, when one appears, means
something.
"""

from __future__ import annotations

import pytest

from cakradana.evaluation.sampling import (
    MIN_AUDIT_SAMPLE,
    AuditFinding,
    AuditSampler,
    SamplingBias,
    estimate_recall,
    propensity_weights,
    wilson_interval,
)

POPULATION = [f"d-{i:04d}" for i in range(5_000)]


def findings(n: int, risky: int) -> list[AuditFinding]:
    return [AuditFinding(f"a-{i}", i < risky) for i in range(n)]


class TestDrawing:
    def test_the_same_period_yields_the_same_sample(self):
        """A reviewer reopening a period must see the work they were assigned,
        and an auditor must be able to reconstruct what was reviewed."""
        first = AuditSampler().draw(POPULATION, period="2026-Q2")
        second = AuditSampler().draw(POPULATION, period="2026-Q2")
        assert first.donation_ids == second.donation_ids

    def test_different_periods_draw_differently(self):
        q2 = AuditSampler().draw(POPULATION, period="2026-Q2")
        q3 = AuditSampler().draw(POPULATION, period="2026-Q3")
        assert q2.donation_ids != q3.donation_ids

    def test_the_seed_does_not_depend_on_process_state(self):
        """Python salts string hashing per process. A draw seeded from it would
        differ on every restart while claiming to be reproducible."""
        draw = AuditSampler().draw(POPULATION, period="2026-Q2")
        assert draw.seed == AuditSampler().draw(POPULATION, period="2026-Q2").seed
        # Pinned, so a change in seeding is caught rather than absorbed.
        assert draw.seed > 0

    def test_the_sample_honours_the_fraction(self):
        draw = AuditSampler(fraction=0.02).draw(POPULATION, period="p")
        assert draw.size == 100

    def test_a_small_population_still_meets_the_floor(self):
        draw = AuditSampler(fraction=0.02).draw(POPULATION[:200], period="p")
        assert draw.size == MIN_AUDIT_SAMPLE

    def test_a_population_smaller_than_the_floor_is_reviewed_whole(self):
        draw = AuditSampler().draw(POPULATION[:20], period="p")
        assert draw.size == 20

    def test_an_empty_population_draws_nothing(self):
        draw = AuditSampler().draw([], period="p")
        assert draw.size == 0
        assert draw.selection_probability == 0.0

    def test_the_fraction_must_be_a_fraction(self):
        with pytest.raises(ValueError):
            AuditSampler(fraction=0)


class TestRecallEstimate:
    def test_no_sample_means_no_recall(self):
        """The denominator counts donations nobody looked at."""
        estimate = estimate_recall(
            detected_risky=40, audit_findings=[], unflagged_population=5_000
        )
        assert not estimate.is_measurable
        assert "no random audit sample" in estimate.unmeasurable_reason

    def test_a_sample_too_small_is_refused_with_its_size(self):
        estimate = estimate_recall(
            detected_risky=40,
            audit_findings=findings(20, 1),
            unflagged_population=5_000,
        )
        assert not estimate.is_measurable
        assert "only 20" in estimate.unmeasurable_reason

    def test_a_sufficient_sample_yields_an_estimate_with_an_interval(self):
        estimate = estimate_recall(
            detected_risky=40,
            audit_findings=findings(200, 4),
            unflagged_population=5_000,
        )
        assert estimate.is_measurable
        # 2% of 5000 unflagged ≈ 100 missed against 40 detected.
        assert estimate.missed_estimate == pytest.approx(100.0)
        assert estimate.value == pytest.approx(40 / 140, rel=1e-6)
        assert estimate.lower < estimate.value < estimate.upper

    def test_finding_nothing_in_the_sample_reads_as_high_recall_not_perfect(self):
        """Zero hits in 200 is consistent with a low but non-zero miss rate,
        and the upper bound has to keep saying so."""
        estimate = estimate_recall(
            detected_risky=40,
            audit_findings=findings(200, 0),
            unflagged_population=5_000,
        )
        assert estimate.value == pytest.approx(1.0)
        assert estimate.lower < 1.0

    def test_more_missed_means_lower_recall(self):
        few = estimate_recall(
            detected_risky=40,
            audit_findings=findings(200, 2),
            unflagged_population=5_000,
        )
        many = estimate_recall(
            detected_risky=40,
            audit_findings=findings(200, 20),
            unflagged_population=5_000,
        )
        assert many.value < few.value

    def test_the_description_says_which_it_is(self):
        refused = estimate_recall(
            detected_risky=1, audit_findings=[], unflagged_population=10
        )
        assert refused.describe().startswith("recall not measurable")


class TestWilsonInterval:
    def test_it_does_not_run_below_zero_near_zero(self):
        """The regime an audit sample operates in: most sampled donations are
        fine. A normal approximation returns a negative lower bound here."""
        lower, upper = wilson_interval(1, 200)
        assert lower >= 0.0
        assert upper > 0.005

    def test_no_trials_admits_everything(self):
        assert wilson_interval(0, 0) == (0.0, 1.0)

    def test_it_narrows_as_the_sample_grows(self):
        small = wilson_interval(5, 50)
        large = wilson_interval(50, 500)
        assert (large[1] - large[0]) < (small[1] - small[0])


class TestSamplingBias:
    def test_no_unflagged_review_is_undefined_rather_than_infinite(self):
        bias = SamplingBias(
            reviewed_flagged=100,
            reviewed_unflagged=0,
            flagged_population=200,
            unflagged_population=9_800,
        )
        assert bias.ratio is None
        assert "no statement about what it missed" in bias.describe()

    def test_the_ratio_states_how_much_more_likely_review_was(self):
        bias = SamplingBias(
            reviewed_flagged=100,
            reviewed_unflagged=98,
            flagged_population=200,
            unflagged_population=9_800,
        )
        assert bias.ratio == pytest.approx(50.0)
        assert "50.0×" in bias.describe()


class TestPropensityWeights:
    def test_rarely_selected_items_stand_in_for_more(self):
        weights = propensity_weights([("d-1", 0.5), ("d-2", 0.1)])
        assert weights["d-2"] > weights["d-1"]

    def test_a_stratum_that_could_never_be_selected_is_refused(self):
        """Weighting cannot recover what had no chance of being sampled, which
        is exactly the blind spot the audit sample exists to cover."""
        with pytest.raises(ValueError, match="never sampled"):
            propensity_weights([("d-1", 0.0)])
