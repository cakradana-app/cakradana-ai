"""Group alerts.

The lane's value depends almost entirely on what these refuse to fire on. A
detector that reports every recipient with many donors reports every popular
candidate, and an analyst who dismisses the first few stops reading the rest.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from cakradana.history import InMemoryDonationStore
from cakradana.lanes.alerts import (
    AlertIndex,
    AlertKind,
    DetectorSettings,
    GroupAlertDetector,
)
from cakradana.lanes.graph import GraphLane
from cakradana.rules import RuleEngine, load_latest
from cakradana.schema import EntityRef, EntityType
from tests.conftest import at, make_donation

WINDOW_END = at(2026, 6, 30)


def smurf_cluster(count: int = 12, amount: int = 190_000_000, spread_days: int = 5):
    """Many thin donors, near-identical amounts, one recipient, days apart."""
    return [
        make_donation(
            donation_id=f"d-smurf-{i}",
            sender=f"e-smurf-{i}",
            receiver="e-party",
            amount_idr=amount + i * 1000,
            occurred=at(2026, 6, 20) + timedelta(days=i % spread_days),
        )
        for i in range(count)
    ]


def grassroots_cluster(count: int = 12):
    """The benign shape that must not fire.

    Donors with their own histories, amounts spread over two orders of
    magnitude, nothing clustered against a limit.
    """
    prior = [
        make_donation(
            donation_id=f"d-prior-{i}",
            sender=f"e-supporter-{i}",
            receiver="e-other-party",
            amount_idr=500_000,
            occurred=at(2025, 3, 1),
        )
        for i in range(count)
    ]
    recent = [
        make_donation(
            donation_id=f"d-organic-{i}",
            sender=f"e-supporter-{i}",
            receiver="e-party",
            amount_idr=50_000 * (i + 1) ** 2,
            occurred=at(2026, 6, 18) + timedelta(days=i),
        )
        for i in range(count)
    ]
    return prior + recent


def detect(donations, **settings):
    store = InMemoryDonationStore(donations)
    detector = GroupAlertDetector(DetectorSettings(**settings))
    return detector.detect(store.knowable_at(WINDOW_END), as_of=WINDOW_END)


class TestFanIn:
    def test_a_thin_homogeneous_burst_is_one_alert(self):
        alerts = detect(smurf_cluster(), threshold_idr=200_000_000)
        fan_in = [a for a in alerts if a.kind is AlertKind.FAN_IN_BURST]
        assert len(fan_in) == 1
        # The point of the lane: twelve donations, one item to review.
        assert len(fan_in[0].subject.donations) == 12

    def test_the_alert_names_the_donors_and_the_window(self):
        alert = detect(smurf_cluster(), threshold_idr=200_000_000)[0]
        assert alert.subject.focus == "e-party"
        assert alert.subject.focus_role == "recipient"
        assert len(alert.subject.counterparties) == 12
        assert alert.subject.window.to == WINDOW_END

    def test_signals_are_reported_alongside_the_score(self):
        """A reader has to be able to disagree with the conclusion while still
        seeing what it rests on."""
        alert = detect(smurf_cluster(), threshold_idr=200_000_000)[0]
        assert alert.signals["distinct_donors"] == 12
        assert alert.signals["donor_thinness_ratio"] == 1.0
        assert alert.signals["amount_cv"] < 0.05
        assert alert.signals["threshold_proximity_ratio"] == 1.0

    def test_grassroots_fundraising_scores_far_lower(self):
        """The central difficulty of this typology. A detector tuned only on
        synthetic smurfing reports perfect precision and fails on contact with
        a successful campaign."""
        smurf = detect(smurf_cluster(), threshold_idr=200_000_000)
        organic = detect(grassroots_cluster(), threshold_idr=200_000_000)

        smurf_score = max(a.score for a in smurf if a.kind is AlertKind.FAN_IN_BURST)
        organic_scores = [
            a.score for a in organic if a.kind is AlertKind.FAN_IN_BURST
        ]
        assert organic_scores, "the shape is present; only its strength differs"
        assert smurf_score > max(organic_scores) * 2

    def test_a_recipient_below_the_donor_floor_produces_nothing(self):
        alerts = detect(smurf_cluster(count=4))
        assert not [a for a in alerts if a.kind is AlertKind.FAN_IN_BURST]

    def test_threshold_proximity_is_unmeasured_rather_than_zero(self):
        """Reporting zero would assert that no amount clusters below the limit,
        which is a finding the detector has no basis for when it was given no
        limit."""
        alert = detect(smurf_cluster())[0]
        assert alert.signals["threshold_proximity_ratio"] is None

    def test_the_comparison_is_measured_not_asserted(self):
        alert = detect(smurf_cluster(), threshold_idr=200_000_000)[0]
        assert "median recipient" in alert.comparison.lower()


class TestProvisionalNodes:
    def test_unresolved_donors_downweight_the_pattern(self):
        """A fan-in of unresolved name variants may be one donor spelled
        inconsistently rather than many donors."""
        resolved = smurf_cluster()
        unresolved = [
            make_donation(
                donation_id=f"d-raw-{i}",
                receiver="e-party",
                amount_idr=190_000_000,
                occurred=at(2026, 6, 21),
            ).model_copy(
                update={
                    "sender_ref": EntityRef(
                        raw_text=f"Budi Santoso {i}", entity_type=EntityType.INDIVIDUAL
                    )
                }
            )
            for i in range(8)
        ]

        mixed = detect(resolved + unresolved, threshold_idr=200_000_000)
        clean = detect(resolved, threshold_idr=200_000_000)
        mixed_alert = next(a for a in mixed if a.kind is AlertKind.FAN_IN_BURST)
        clean_alert = next(a for a in clean if a.kind is AlertKind.FAN_IN_BURST)

        assert mixed_alert.provisional_node_ratio > 0
        assert mixed_alert.score < clean_alert.score

    def test_unresolved_donors_are_not_counted_as_distinct_donors(self):
        unresolved = [
            make_donation(
                donation_id=f"d-raw-{i}", receiver="e-party", amount_idr=190_000_000
            ).model_copy(
                update={
                    "sender_ref": EntityRef(
                        raw_text="Budi", entity_type=EntityType.INDIVIDUAL
                    )
                }
            )
            for i in range(20)
        ]
        assert not [
            a for a in detect(unresolved) if a.kind is AlertKind.FAN_IN_BURST
        ]


class TestFanOut:
    def test_one_donor_reaching_many_recipients(self):
        donations = [
            make_donation(
                donation_id=f"d-out-{i}",
                sender="e-conduit",
                receiver=f"e-candidate-{i}",
                amount_idr=100_000_000,
                occurred=at(2026, 6, 20) + timedelta(hours=i),
            )
            for i in range(8)
        ]
        alert = next(
            a for a in detect(donations) if a.kind is AlertKind.FAN_OUT
        )
        assert alert.subject.focus == "e-conduit"
        assert alert.signals["distinct_recipients"] == 8
        # Equal outflows read as one total being partitioned.
        assert alert.signals["amount_partition_regularity"] == 1.0


class TestLayeringChains:
    def test_a_chain_of_three_is_reported_with_its_depth(self):
        donations = [
            make_donation(
                donation_id="d-leg-1",
                sender="e-origin",
                receiver="e-mid-1",
                amount_idr=1_000_000_000,
                occurred=at(2026, 6, 20),
            ),
            make_donation(
                donation_id="d-leg-2",
                sender="e-mid-1",
                receiver="e-mid-2",
                amount_idr=950_000_000,
                occurred=at(2026, 6, 22),
            ),
            make_donation(
                donation_id="d-leg-3",
                sender="e-mid-2",
                receiver="e-party",
                amount_idr=900_000_000,
                occurred=at(2026, 6, 24),
            ),
        ]
        chains = [a for a in detect(donations) if a.kind is AlertKind.LAYERING_CHAIN]
        assert chains
        longest = max(chains, key=lambda a: a.signals["hops"])
        assert longest.signals["hops"] == 3
        assert longest.signals["amount_attenuation"] == pytest.approx(0.9)

    def test_an_unrelated_later_donation_is_not_a_chain_leg(self):
        """An entity that receives and then gives an unrelated amount months
        later is not a conduit; it is an entity that does two things."""
        donations = [
            make_donation(
                donation_id="d-in",
                sender="e-origin",
                receiver="e-mid",
                amount_idr=1_000_000_000,
                occurred=at(2026, 6, 1),
            ),
            make_donation(
                donation_id="d-out",
                sender="e-mid",
                receiver="e-party",
                amount_idr=5_000_000,
                occurred=at(2026, 6, 28),
            ),
        ]
        assert not [
            a for a in detect(donations) if a.kind is AlertKind.LAYERING_CHAIN
        ]


class TestIdentity:
    def test_the_same_cluster_detected_twice_is_the_same_alert(self):
        """An analyst who dispositioned a cluster yesterday must not be shown
        it again today under a new identifier."""
        first = detect(smurf_cluster(), threshold_idr=200_000_000)
        second = detect(smurf_cluster(), threshold_idr=200_000_000)
        assert [a.alert_id for a in first] == [a.alert_id for a in second]

    def test_a_changed_membership_is_a_different_alert(self):
        small = detect(smurf_cluster(count=10), threshold_idr=200_000_000)[0]
        large = detect(smurf_cluster(count=12), threshold_idr=200_000_000)[0]
        assert small.alert_id != large.alert_id


class TestPointInTime:
    def test_a_cluster_is_invisible_before_its_members_are_recorded(self):
        """Detection using donations recorded later is not detection that was
        possible then, and scoring history with it is leakage in a form that is
        hard to notice."""
        store = InMemoryDonationStore(smurf_cluster())
        detector = GroupAlertDetector(DetectorSettings(threshold_idr=200_000_000))
        early = detector.detect(
            store.knowable_at(at(2026, 6, 19)), as_of=at(2026, 6, 19)
        )
        assert not early


class TestLaneIntegration:
    @pytest.fixture
    def context(self):
        donations = smurf_cluster()
        store = InMemoryDonationStore(donations)
        engine = RuleEngine(load_latest(), require_verified_citations=False)

        from cakradana.features import FeatureService

        features = FeatureService(load_latest())
        target = donations[0]
        view = store.knowable_at(WINDOW_END)
        ctx = features.context_for(target, view)
        return engine.evaluate(target, view), ctx, store

    def test_a_member_donation_inherits_the_cluster_score(self, context):
        evaluation, ctx, store = context
        alerts = GroupAlertDetector(
            DetectorSettings(threshold_idr=200_000_000)
        ).detect(store.knowable_at(WINDOW_END), as_of=WINDOW_END)

        bare = GraphLane().evaluate(evaluation, ctx)
        informed = GraphLane(AlertIndex(alerts)).evaluate(evaluation, ctx)
        assert informed.contribution >= bare.contribution
        assert informed.reasons

    def test_the_reason_points_at_the_group_not_the_donation(self, context):
        """An analyst following the evidence should arrive at the pattern
        rather than back at the single payment, which justifies nothing on its
        own."""
        evaluation, ctx, store = context
        alerts = GroupAlertDetector(
            DetectorSettings(threshold_idr=200_000_000)
        ).detect(store.knowable_at(WINDOW_END), as_of=WINDOW_END)
        result = GraphLane(AlertIndex(alerts)).evaluate(evaluation, ctx)

        group_reason = next(
            r for r in result.reasons if r.code == str(AlertKind.FAN_IN_BURST)
        )
        assert group_reason.evidence_ref.startswith("cluster:")
        assert "one of 12" in group_reason.statement

    def test_a_donation_outside_every_cluster_is_unaffected(self, context):
        evaluation, ctx, _store = context
        lonely = AlertIndex(
            detect(
                [
                    make_donation(
                        donation_id=f"d-elsewhere-{i}",
                        sender=f"e-elsewhere-{i}",
                        receiver="e-unrelated",
                        amount_idr=190_000_000,
                        occurred=at(2026, 6, 21),
                    )
                    for i in range(12)
                ]
            )
        )
        assert not lonely.covering(ctx.donation.donation_id)
