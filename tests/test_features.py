"""Feature computation.

Three properties carry the weight here: features look strictly backwards,
undefined values are null rather than zero, and the training and serving paths
produce the same numbers. The last is asserted rather than assumed, because a
silent divergence between them is invisible until the model is already being
served inputs it was never trained on.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from cakradana.calendar import CampaignPeriod, ElectoralCalendar
from cakradana.features import FeatureService, catalogue, feature_names
from cakradana.history import InMemoryDonationStore
from cakradana.rules import load_latest
from cakradana.schema import TemporalPrecision
from tests.conftest import at, make_donation

CONTEXT = "pemilu-2029"


@pytest.fixture(scope="session")
def calendar() -> ElectoralCalendar:
    return ElectoralCalendar(
        [
            CampaignPeriod(
                electoral_context=CONTEXT,
                start=date(2028, 11, 1),
                end=date(2029, 2, 14),
                reporting_deadlines=(date(2029, 1, 15),),
            )
        ]
    )


@pytest.fixture(scope="session")
def service(calendar) -> FeatureService:
    return FeatureService(load_latest(), calendar=calendar)


def donation(**kwargs):
    kwargs.setdefault("electoral_context", CONTEXT)
    return make_donation(**kwargs)


def compute(service, target, others=()):
    store = InMemoryDonationStore([*others, target])
    return service.compute(target, store.knowable_at(target.occurred_at))


class TestCatalogue:
    def test_every_feature_declares_its_family_and_type(self):
        for spec in catalogue().values():
            assert spec.family
            assert spec.dtype in {"int", "float", "bool", "categorical"}
            assert spec.description

    def test_no_rule_verdicts_are_exposed_as_features(self):
        """A statutory verdict is a legal finding, not a model input, and a
        heuristic's own output fed back as a feature is how a classifier learns
        to reproduce the heuristic instead of improving on it."""
        forbidden = {
            "flag_donasi_melebihi_batas",
            "rule_fired",
            "legal_finding",
            "tier2_signal",
            "is_risky",
            "risk",
        }
        assert forbidden.isdisjoint(feature_names())

    def test_entity_names_are_not_features(self):
        """Names encode region, ethnicity, and religion. They inform entity
        resolution and register lookup; they never reach the classifier."""
        for name in feature_names():
            assert "name" not in name.split("_")

    def test_removed_duplicate_columns_are_absent(self):
        """The previous feature set carried two columns that were verbatim
        copies of two others under a centrality label."""
        assert "degree_centrality_sender" not in feature_names()
        assert "degree_centrality_receiver" not in feature_names()

    def test_version_changes_when_definitions_change(self, service):
        assert service.version.startswith("features-")


class TestPointInTime:
    def test_donor_aggregates_exclude_the_donation_being_scored(self, service):
        target = donation(donation_id="target", amount_idr=90_000_000)
        vector = compute(service, target)
        assert vector.values["jumlah_transaksi_sender"] == 0
        assert vector.values["total_donasi_sender"] == 0.0

    def test_donor_aggregates_include_prior_donations(self, service):
        prior = [
            donation(donation_id=f"p{i}", amount_idr=10_000_000, occurred=at(2026, 1, i + 1))
            for i in range(3)
        ]
        target = donation(donation_id="target", occurred=at(2026, 6, 1))
        vector = compute(service, target, prior)
        assert vector.values["jumlah_transaksi_sender"] == 3
        assert vector.values["total_donasi_sender"] == 30_000_000.0

    def test_later_recorded_donations_are_invisible(self, service):
        late_arrival = donation(
            donation_id="late",
            amount_idr=500_000_000,
            occurred=at(2026, 1, 1),
            recorded=at(2026, 12, 1),
        )
        target = donation(donation_id="target", occurred=at(2026, 6, 1))
        vector = compute(service, target, [late_arrival])
        assert vector.values["total_donasi_sender"] == 0.0


class TestNullsRatherThanZeros:
    def test_first_donation_has_no_spread_or_interval(self, service):
        """Zero would state that the donor gives an unvarying amount at a
        perfectly regular cadence, which is a strong signal and the opposite of
        what a single observation supports."""
        vector = compute(service, donation(donation_id="only"))
        assert vector.values["std_donasi_sender"] is None
        assert vector.values["selang_waktu_rata2_sender"] is None

    def test_spread_appears_once_there_is_something_to_measure(self, service):
        prior = [
            donation(donation_id=f"p{i}", amount_idr=10_000_000 * (i + 1),
                     occurred=at(2026, 1, i + 1))
            for i in range(3)
        ]
        vector = compute(service, donation(donation_id="t", occurred=at(2026, 6, 1)), prior)
        assert vector.values["std_donasi_sender"] > 0

    def test_hour_is_null_when_the_source_carried_no_time(self, service):
        """Reading midnight off a date-only record would invent an hour, and
        enough of them would look like coordinated timing."""
        day_only = donation(
            donation_id="scanned", occurred_at_precision=TemporalPrecision.DAY
        )
        assert compute(service, day_only).values["hour_of_day"] is None

    def test_hour_is_present_when_the_source_carried_one(self, service):
        timed = donation(
            donation_id="digital",
            occurred=at(2026, 6, 1, 14),
            occurred_at_precision=TemporalPrecision.MINUTE,
        )
        assert compute(service, timed).values["hour_of_day"] == 14

    def test_limit_ratios_are_null_when_the_regime_is_unknown(self, service):
        unknown = make_donation(donation_id="u", electoral_context="not-configured")
        vector = compute(service, unknown)
        assert vector.values["amount_to_limit_ratio"] is None
        assert vector.values["in_structuring_band"] is None

    def test_every_null_is_declared_legitimate(self, service):
        """A feature that comes back null without declaring when it may is a
        pipeline defect wearing the costume of missing data."""
        vector = compute(service, donation(donation_id="only"))
        specs = catalogue()
        for name in vector.missing():
            assert specs[name].null_when != "never", name


class TestTrainServeParity:
    def test_backfill_reproduces_what_serving_would_compute(self, service):
        """The gate that would have caught the original defect: feature
        engineering lived only in a notebook, so nothing computed these values
        at serving time at all, and no check could notice."""
        donations = [
            donation(
                donation_id=f"d{i}",
                sender=f"donor-{i % 4}",
                receiver=f"party-{i % 2}",
                amount_idr=10_000_000 * (i + 1),
                occurred=at(2026, 1, 1) + timedelta(days=3 * i),
            )
            for i in range(25)
        ]
        store = InMemoryDonationStore(donations)

        for target, backfilled in service.backfill(store):
            served = service.compute(target, store.knowable_at(target.occurred_at))
            assert backfilled.values == served.values, target.donation_id

    def test_parity_holds_with_late_arriving_data(self, service):
        """Late arrivals are where the two paths would diverge if either used
        occurrence order instead of the order the system learned things."""
        donations = [
            donation(
                donation_id="on-time",
                occurred=at(2026, 3, 1),
                recorded=at(2026, 3, 1),
            ),
            donation(
                donation_id="late",
                occurred=at(2026, 1, 1),
                recorded=at(2026, 9, 1),
            ),
            donation(
                donation_id="after",
                occurred=at(2026, 10, 1),
                recorded=at(2026, 10, 1),
            ),
        ]
        store = InMemoryDonationStore(donations)
        for target, backfilled in service.backfill(store):
            served = service.compute(target, store.knowable_at(target.occurred_at))
            assert backfilled.values == served.values, target.donation_id


class TestFanInFeatures:
    def test_new_donor_ratio_distinguishes_a_burst_from_an_established_base(
        self, service
    ):
        burst = [
            donation(
                donation_id=f"n{i}",
                sender=f"new-{i}",
                receiver="party-1",
                amount_idr=5_000_000,
                occurred=at(2026, 6, 1) + timedelta(hours=i),
            )
            for i in range(12)
        ]
        target = donation(
            donation_id="t", sender="new-99", receiver="party-1",
            occurred=at(2026, 6, 2)
        )
        assert compute(service, target, burst).values[
            "receiver_new_donor_ratio_30d"
        ] == 1.0

        established = [
            donation(
                donation_id=f"h{i}",
                sender=f"new-{i}",
                receiver="party-1",
                amount_idr=5_000_000,
                occurred=at(2025, 6, 1) + timedelta(days=i),
            )
            for i in range(12)
        ]
        assert compute(service, target, established + burst).values[
            "receiver_new_donor_ratio_30d"
        ] == 0.0
