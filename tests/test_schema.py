"""Invariants the canonical schema must enforce at construction time.

Each test here corresponds to a defect class that silently corrupted results
before the schema existed, so they assert rejection rather than behaviour.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from cakradana.schema import (
    Channel,
    Donation,
    Entity,
    EntityRef,
    EntityType,
    FieldProvenance,
    Label,
    LabelSource,
    LabelValue,
    Provenance,
    TemporalPrecision,
)
from tests.conftest import at, make_donation


class TestDonationTimestamps:
    def test_naive_timestamps_are_rejected(self):
        with pytest.raises(ValidationError, match="timezone"):
            make_donation(occurred=datetime(2026, 6, 1))

    def test_recorded_before_occurred_is_rejected(self):
        """The system cannot have learned of a donation before it happened.

        Permitting this would let a backdated record contribute to point-in-time
        aggregates that predate its own existence.
        """
        with pytest.raises(ValidationError, match="precedes occurred_at"):
            make_donation(occurred=at(2026, 6, 10), recorded=at(2026, 6, 1))

    def test_recorded_may_lag_occurred(self):
        d = make_donation(occurred=at(2026, 1, 5), recorded=at(2026, 6, 30))
        assert d.recorded_at > d.occurred_at


class TestDonationAmounts:
    @pytest.mark.parametrize("amount", [0, -1, -95_000_000])
    def test_non_positive_amounts_are_rejected(self, amount):
        with pytest.raises(ValidationError):
            make_donation(amount_idr=amount)

    def test_raw_amount_is_retained_alongside_the_parsed_value(self):
        d = make_donation(amount_idr=100_000_000, amount_raw="Rp100.000.000")
        assert d.amount_raw == "Rp100.000.000"


class TestDedupKey:
    def test_same_donation_at_different_precisions_collides(self):
        """A scanned form and a digital submission of one donation must not
        both be counted: double-counting inflates cumulative totals and can
        manufacture a statutory finding that did not occur."""
        scanned = make_donation(
            donation_id="d-scan",
            occurred=at(2026, 6, 1),
            occurred_at_precision=TemporalPrecision.DAY,
            channel=Channel.PAPER_FORM,
        )
        digital = make_donation(
            donation_id="d-digital",
            occurred=at(2026, 6, 1, 14, 30),
            occurred_at_precision=TemporalPrecision.DAY,
            channel=Channel.DIGITAL_FORM,
        )
        assert scanned.dedup_key() == digital.dedup_key()

    def test_different_amounts_do_not_collide(self):
        a = make_donation(amount_idr=1_000_000)
        b = make_donation(amount_idr=2_000_000)
        assert a.dedup_key() != b.dedup_key()

    def test_unresolved_senders_do_not_collide_with_resolved_ones(self):
        resolved = make_donation()
        unresolved = Donation(
            donation_id="d-2",
            sender_ref=EntityRef(raw_text="e-sender"),
            receiver_ref=EntityRef(entity_id="e-receiver"),
            amount_idr=10_000_000,
            occurred_at=at(2026, 6, 1),
            recorded_at=at(2026, 6, 1),
            channel=Channel.WEB_SCRAPE,
        )
        assert resolved.dedup_key() != unresolved.dedup_key()


class TestEntityRef:
    def test_a_reference_needs_an_identity_or_raw_text(self):
        with pytest.raises(ValidationError):
            EntityRef()

    def test_unresolved_reference_refuses_to_produce_a_grouping_key(self):
        """Aggregating by raw name would let a donor split across name variants
        form separate groups and evade cumulative limits entirely."""
        ref = EntityRef(raw_text="Budi Santoso")
        assert not ref.is_resolved
        with pytest.raises(ValueError, match="no aggregation key"):
            _ = ref.key

    def test_resolved_reference_keys_on_entity_id(self):
        assert EntityRef(entity_id="e-1", raw_text="Budi").key == "e-1"


class TestProvenance:
    def test_human_correction_requires_an_actor(self):
        with pytest.raises(ValidationError, match="actor"):
            FieldProvenance(provenance=Provenance.HUMAN_CORRECTED)

    def test_minimum_extraction_confidence_is_the_weakest_field(self):
        d = make_donation(
            provenance={
                "amount_idr": FieldProvenance(
                    provenance=Provenance.EXTRACTED, confidence=0.93
                ),
                "occurred_at": FieldProvenance(
                    provenance=Provenance.EXTRACTED, confidence=0.62
                ),
            }
        )
        assert d.extraction_confidence_min == 0.62

    def test_confidence_is_none_when_nothing_carries_one(self, donation):
        assert donation.extraction_confidence_min is None


class TestTemporalPrecision:
    def test_day_precision_has_no_time_of_day(self):
        """Features keyed on hour of day must refuse to compute rather than
        read midnight off a date-only source and manufacture clustering."""
        assert not TemporalPrecision.DAY.has_time_of_day
        assert TemporalPrecision.HOUR.has_time_of_day


class TestEntity:
    def test_seen_range_must_be_ordered(self):
        with pytest.raises(ValidationError, match="precedes first_seen"):
            Entity(
                entity_id="e-1",
                canonical_name="Budi",
                first_seen=at(2026, 6, 1),
                last_seen=at(2026, 1, 1),
            )

    def test_unknown_is_a_valid_entity_type(self):
        assert Entity(entity_id="e-1", canonical_name="?").entity_type is EntityType.UNKNOWN


class TestLabels:
    def test_recipient_confirmation_cannot_assert_a_clean_donation(self):
        """Confirmation establishes that a transaction occurred, not that it is
        legitimate. A smurfed donation is genuinely received and its recipient
        confirms it truthfully."""
        with pytest.raises(ValidationError, match="occurrence"):
            Label(
                label_id="l-1",
                donation_id="d-1",
                donation_version=1,
                value=LabelValue.NOT_RISKY,
                source=LabelSource.RECIPIENT_CONFIRMATION,
                weight=0.7,
                created_at=at(2026, 6, 1),
            )

    def test_recipient_confirmation_is_recorded_as_indeterminate(self):
        label = Label(
            label_id="l-1",
            donation_id="d-1",
            donation_version=1,
            value=LabelValue.INDETERMINATE,
            source=LabelSource.RECIPIENT_CONFIRMATION,
            weight=0.7,
            created_at=at(2026, 6, 1),
        )
        assert not label.is_human

    def test_analyst_dispositions_count_as_human_labels(self):
        label = Label(
            label_id="l-2",
            donation_id="d-1",
            donation_version=1,
            value=LabelValue.RISKY,
            source=LabelSource.ANALYST_DISPOSITION,
            weight=0.9,
            created_at=at(2026, 6, 1),
        )
        assert label.is_human

    def test_heuristic_labels_weigh_less_than_adjudicated_ones(self):
        assert Label.default_weight_for(
            LabelSource.RULE_TIER2
        ) < Label.default_weight_for(LabelSource.DISPUTE_OUTCOME)
