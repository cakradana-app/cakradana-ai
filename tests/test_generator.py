"""Synthetic data generation.

The central test is that every typology the generator labels is actually
present in the structure it produces. That check is the guard against the
failure it replaces: the previous data labelled patterns it never encoded, so
a model trained on it was fitting noise across most of its positive class while
reporting healthy-looking scores throughout.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta

import pytest

from cakradana.data import (
    ALL_TYPOLOGIES,
    GENERATOR_VERSION,
    GeneratorConfig,
    assert_acceptable,
    check,
    generate,
)
from cakradana.data.generator import INDIVIDUAL_PARTY_LIMIT, T_SMURFING


def _spread(values) -> float:
    """Coefficient of variation."""
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return variance**0.5 / mean


# A smaller dataset keeps the suite quick while preserving every structure.
SMALL = GeneratorConfig(
    seed=4242,
    n_legitimate_donors=250,
    n_recipients=6,
    n_background_donations=1500,
    n_grassroots_campaigns=3,
)


@pytest.fixture(scope="module")
def dataset():
    return generate(SMALL)


class TestTypologyStructure:
    @pytest.mark.parametrize("seed", [1, 7, 99, 20260816])
    def test_every_typology_is_recoverable_by_its_defining_signal(self, seed):
        """A row labelled as donation splitting that lacks converging donors is
        not donation splitting; it is a mislabelled ordinary donation."""
        results = assert_acceptable(generate(replace(SMALL, seed=seed)))
        assert {r.typology for r in results} == set(ALL_TYPOLOGIES)

    def test_acceptance_reports_recall_per_typology(self, dataset):
        for result in check(dataset):
            assert result.recall >= 0.8, result.describe()


class TestBaseRate:
    def test_risky_donations_are_a_small_minority(self, dataset):
        """A balanced dataset makes class weighting inert and produces
        precision estimates that do not survive a realistic prevalence."""
        rate = dataset.manifest["observed_risky_rate"]
        assert 0.01 <= rate <= 0.06

    def test_the_configured_rate_is_recorded_alongside_the_observed_one(self, dataset):
        assert dataset.manifest["configured_risky_rate"] == SMALL.risky_rate


class TestNegatives:
    def test_benign_fan_in_exists(self, dataset):
        """Without genuine fundraising surges, any fan-in detector scores
        perfectly here and collapses on real data."""
        supporters = [e for e in dataset.entities if e.startswith("supporter")]
        assert len(supporters) > 50

    def test_benign_fan_in_is_comparable_in_size_to_a_split_cohort(self, dataset):
        """The negatives only do their job if a naive detector would actually
        have to distinguish them. Grassroots surges converge as many donors on
        a recipient as a split contribution does; what separates them is that
        real supporters choose varied amounts."""
        from collections import defaultdict

        by_recipient = defaultdict(set)
        for d in dataset.donations:
            if d.sender_ref.entity_id.startswith("supporter"):
                by_recipient[d.receiver_ref.entity_id].add(d.sender_ref.entity_id)
        assert max(len(v) for v in by_recipient.values()) >= 15

        grassroots_amounts = [
            d.amount_idr
            for d in dataset.donations
            if d.sender_ref.entity_id.startswith("supporter")
        ]
        split_amounts = [
            d.amount_idr
            for d in dataset.donations
            if dataset.truth.get(d.donation_id) == T_SMURFING
        ]
        assert _spread(grassroots_amounts) > _spread(split_amounts) * 2

    def test_amounts_are_heavy_tailed_not_uniform(self, dataset):
        amounts = sorted(d.amount_idr for d in dataset.donations)
        median = amounts[len(amounts) // 2]
        largest = amounts[-1]
        assert largest > median * 10

    def test_donations_span_the_configured_period(self, dataset):
        dates = {d.occurred_at.date() for d in dataset.donations}
        assert min(dates) >= SMALL.period_start
        assert len(dates) > 200


class TestProvenanceAndArrival:
    def test_scraped_records_arrive_later_than_they_occurred(self, dataset):
        lagged = [d for d in dataset.donations if d.recorded_at > d.occurred_at]
        assert lagged, "no late-arriving data, so point-in-time handling is untested"

    def test_scanned_records_carry_extraction_confidence(self, dataset):
        from cakradana.schema import Channel

        scanned = [d for d in dataset.donations if d.channel is Channel.PAPER_FORM]
        assert scanned
        assert all(d.extraction_confidence_min is not None for d in scanned)

    def test_scanned_records_have_no_time_of_day(self, dataset):
        from cakradana.schema import Channel

        for d in dataset.donations:
            if d.channel is Channel.PAPER_FORM:
                assert not d.occurred_at_precision.has_time_of_day


class TestIllegalSourceSignal:
    def test_the_signal_lives_on_the_entity_not_the_name(self, dataset):
        """The previous data encoded this typology entirely in the donor's
        name, then dropped the name before training, leaving nothing to learn
        from."""
        prohibited = {
            e.entity_id
            for e in dataset.entities.values()
            if "prohibited_source" in e.registers
        }
        assert prohibited
        flagged = {
            d.sender_ref.entity_id
            for d, t in ((d, dataset.truth.get(d.donation_id)) for d in dataset.donations)
            if t == "T-05"
        }
        assert flagged <= prohibited

    def test_the_register_is_supplied_with_the_dataset(self, dataset):
        from cakradana.registers import RegisterSet

        lookup = dataset.registers.get(RegisterSet.PROHIBITED_SOURCE)
        assert len(lookup) > 0


class TestCumulativePattern:
    def test_each_cumulative_donation_is_lawful_on_its_own(self, dataset):
        """The whole point of the pattern: every payment passes a
        single-transaction check while the total does not."""
        for donation in dataset.donations:
            if dataset.truth.get(donation.donation_id) == "T-02":
                assert donation.amount_idr <= INDIVIDUAL_PARTY_LIMIT


class TestReproducibility:
    def test_the_same_seed_produces_the_same_data(self):
        a, b = generate(SMALL), generate(SMALL)
        assert [d.donation_id for d in a.donations] == [d.donation_id for d in b.donations]
        assert [d.amount_idr for d in a.donations] == [d.amount_idr for d in b.donations]
        assert a.truth == b.truth

    def test_a_different_seed_produces_different_data(self):
        a = generate(SMALL)
        b = generate(replace(SMALL, seed=SMALL.seed + 1))
        assert [d.amount_idr for d in a.donations] != [d.amount_idr for d in b.donations]

    def test_the_manifest_records_how_the_data_was_made(self, dataset):
        assert dataset.manifest["generator_version"] == GENERATOR_VERSION
        assert dataset.manifest["seed"] == SMALL.seed
        assert set(dataset.manifest["typology_counts"]) == set(ALL_TYPOLOGIES)


class TestRetirement:
    """Synthetic data does not age on its own, which is why it is stamped.

    A generated file neither expires nor contradicts itself, so it survives in
    a directory, gets copied, and is eventually cited by somebody who has
    forgotten where it came from.
    """

    def test_the_manifest_says_the_data_is_synthetic_before_anything_else(self):
        dataset = generate(SMALL)
        assert next(iter(dataset.manifest)) == "synthetic"
        assert dataset.manifest["synthetic"] is True

    def test_a_generated_dataset_carries_its_retirement_date(self):
        dataset = generate(replace(SMALL, generated_on=date(2026, 1, 1)))
        assert dataset.generated_on == date(2026, 1, 1)
        assert dataset.retire_on == date(2026, 1, 1) + timedelta(
            days=SMALL.retire_after_days
        )

    def test_a_fresh_dataset_is_not_retired(self):
        dataset = generate(replace(SMALL, generated_on=date(2026, 1, 1)))
        assert dataset.is_retired(date(2026, 3, 1)) is False

    def test_a_dataset_past_its_date_is_retired(self):
        dataset = generate(replace(SMALL, generated_on=date(2026, 1, 1)))
        assert dataset.is_retired(date(2027, 1, 1)) is True

    def test_the_boundary_day_counts_as_retired(self):
        """Fit "until" the date, not through it."""
        dataset = generate(replace(SMALL, generated_on=date(2026, 1, 1)))
        assert dataset.is_retired(dataset.retire_on) is True

    def test_the_notice_states_the_date_while_the_data_is_still_fit(self):
        """The reader who needs to know it expired is usually holding a copy
        made before it did."""
        dataset = generate(replace(SMALL, generated_on=date(2026, 1, 1)))
        notice = dataset.retirement_notice(date(2026, 2, 1))
        assert "fit to use until" in notice
        assert "no real donation" in notice

    def test_the_notice_says_what_to_do_once_it_has_expired(self):
        dataset = generate(replace(SMALL, generated_on=date(2026, 1, 1)))
        notice = dataset.retirement_notice(date(2027, 1, 1))
        assert "regenerate" in notice
        assert str(dataset.retire_on) in notice

    def test_the_retirement_window_is_configurable(self):
        dataset = generate(
            replace(SMALL, generated_on=date(2026, 1, 1), retire_after_days=30)
        )
        assert dataset.retire_on == date(2026, 1, 31)
