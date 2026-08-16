"""The external reputation lane.

The only lane whose default is not to run. It reasons about what has been
written about a person rather than about transactions the system holds records
of, so it can be wrong in ways the others cannot: a donor reported on and
cleared, a donor sharing a name with someone in the news, and a donor targeted
by hostile coverage are indistinguishable to it.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from cakradana.calendar import CampaignPeriod, ElectoralCalendar
from cakradana.history import InMemoryDonationStore
from cakradana.lanes.reputation import (
    MIN_INDEPENDENT_SOURCES,
    MIN_MATCH_CONFIDENCE,
    CoverageIndex,
    CoverageItem,
    OperatingConditions,
    ReputationLane,
)
from cakradana.rules import RuleEngine, load_latest
from cakradana.scoring.result import Lane
from tests.conftest import WIB, at, make_donation

CONTEXT = "pemilu-2029"

ALL_MET = OperatingConditions(
    defamation_review_completed=True,
    source_list_published=True,
    matching_accuracy_measured=True,
    subject_access_route_exists=True,
    retraction_handling_implemented=True,
    named_owner="compliance@example.org",
    lift_measured=True,
)


def coverage(**overrides) -> CoverageItem:
    defaults = {
        "entity_id": "e-sender",
        "source": "Kompas",
        "published_at": at(2026, 5, 1),
        "headline": "Reported",
        "url": "https://example.org/a",
        "match_confidence": 0.99,
        "stage": "allegation",
    }
    return CoverageItem(**{**defaults, **overrides})


def index(*items: CoverageItem) -> CoverageIndex:
    idx = CoverageIndex()
    for item in items:
        idx.add(item)
    return idx


@pytest.fixture
def context():
    calendar = ElectoralCalendar(
        [CampaignPeriod(electoral_context=CONTEXT, start=date(2026, 9, 1), end=date(2026, 11, 24))]
    )
    engine = RuleEngine(load_latest(), calendar=calendar, require_verified_citations=False)
    donation = make_donation(occurred=at(2026, 6, 1), electoral_context=CONTEXT)
    store = InMemoryDonationStore([donation])
    view = store.knowable_at(donation.occurred_at)
    from cakradana.features import FeatureService

    features = FeatureService(load_latest(), calendar=calendar)
    ctx = features.context_for(donation, view)
    evaluation = engine.evaluate(donation, view)
    vector = features.compute_from_context(ctx)
    return evaluation, ctx, vector


class TestOperatingGate:
    def test_the_lane_does_not_run_by_default(self, context):
        """Switching it on should be a deliberate act with a record, not a
        config value someone flips while looking at something else."""
        lane = ReputationLane(index(coverage()))
        result = lane.evaluate(*context)
        assert not result.available
        assert "not operating" in result.unavailable_reason

    def test_the_reason_names_what_is_missing(self, context):
        lane = ReputationLane(index(coverage()))
        reason = lane.evaluate(*context).unavailable_reason
        assert "defamation exposure review" in reason
        assert "accountable" in reason

    def test_every_condition_defaults_to_unmet(self):
        assert len(OperatingConditions().unmet()) == 7
        assert not OperatingConditions().satisfied

    def test_a_partially_configured_gate_still_refuses(self, context):
        partial = OperatingConditions(
            defamation_review_completed=True, source_list_published=True
        )
        lane = ReputationLane(index(coverage(), coverage(source="Tempo")), partial)
        assert not lane.evaluate(*context).available

    def test_the_lane_runs_once_every_condition_is_met(self, context):
        lane = ReputationLane(
            index(coverage(), coverage(source="Tempo", url="https://example.org/b")),
            ALL_MET,
        )
        result = lane.evaluate(*context)
        assert result.available
        assert result.reasons


class TestAttribution:
    def test_a_weak_name_match_is_not_attributed(self, context):
        """Name collision is the failure that turns this lane into a machine
        for defaming people who share a name with someone in the news."""
        lane = ReputationLane(
            index(
                coverage(match_confidence=MIN_MATCH_CONFIDENCE - 0.1),
                coverage(source="Tempo", match_confidence=MIN_MATCH_CONFIDENCE - 0.1),
            ),
            ALL_MET,
        )
        result = lane.evaluate(*context)
        assert not result.available
        assert "no coverage matched" in result.unavailable_reason

    def test_a_single_source_is_not_enough(self, context):
        """One outlet repeating a claim is one claim, however many times it is
        republished."""
        lane = ReputationLane(
            index(coverage(), coverage(url="https://example.org/b")), ALL_MET
        )
        result = lane.evaluate(*context)
        assert not result.available
        assert f"{MIN_INDEPENDENT_SOURCES} independent sources" in result.unavailable_reason

    def test_an_unresolved_donor_gets_no_coverage_attributed(self, context):
        from cakradana.schema import Channel, Donation, EntityRef, EntityType

        evaluation, ctx, vector = context
        ctx.donation = Donation(
            donation_id="d-raw",
            sender_ref=EntityRef(raw_text="Budi Santoso", entity_type=EntityType.INDIVIDUAL),
            receiver_ref=EntityRef(entity_id="party-1", entity_type=EntityType.POLITICAL_PARTY),
            amount_idr=1_000_000,
            occurred_at=at(2026, 6, 1),
            recorded_at=at(2026, 6, 1),
            channel=Channel.WEB_SCRAPE,
        )
        lane = ReputationLane(index(coverage(), coverage(source="Tempo")), ALL_MET)
        result = lane.evaluate(evaluation, ctx, vector)
        assert not result.available
        assert "must not be attributed to a name" in result.unavailable_reason


class TestCoverageLifecycle:
    def test_coverage_published_after_the_donation_is_invisible(self):
        """Admitting it would make a historical score impossible to reproduce:
        the scorer would have used something that did not yet exist."""
        idx = index(coverage(published_at=at(2026, 8, 1)))
        assert idx.about("e-sender", as_of=at(2026, 6, 1)) == []

    def test_stale_coverage_expires(self):
        """A story from a decade ago says little about a donation made last
        week, and letting it persist means a person never stops being its
        subject."""
        idx = index(coverage(published_at=at(2020, 1, 1)))
        assert idx.about("e-sender", as_of=at(2026, 6, 1)) == []

    def test_retracted_coverage_stops_counting(self):
        idx = index(coverage(), coverage(source="Tempo", url="https://example.org/b"))
        assert idx.retract("https://example.org/a") == 1
        remaining = idx.about("e-sender", as_of=at(2026, 6, 1))
        assert [item.source for item in remaining] == ["Tempo"]

    def test_retraction_excludes_rather_than_deletes(self):
        """A score recorded earlier must still be explicable by what was known
        at the time."""
        idx = index(coverage())
        idx.retract("https://example.org/a")
        assert len(idx.items) == 1
        assert idx.items[0].retracted


class TestWhatItClaims:
    def test_the_statement_describes_coverage_not_conduct(self, context):
        """"Reported as under investigation" and "has been investigated" are
        different claims, and only the first is one this lane can support."""
        lane = ReputationLane(
            index(coverage(), coverage(source="Tempo", url="https://example.org/b")),
            ALL_MET,
        )
        reason = lane.evaluate(*context).reasons[0]
        assert "what has been written" in reason.statement
        assert "not a finding about what the donor did" in reason.statement

    def test_it_contributes_the_smallest_share_of_any_lane(self, context):
        lane = ReputationLane(
            index(
                coverage(stage="adjudicated"),
                coverage(source="Tempo", stage="adjudicated", url="https://example.org/b"),
                coverage(source="Detik", stage="adjudicated", url="https://example.org/c"),
                coverage(source="Antara", stage="adjudicated", url="https://example.org/d"),
            ),
            ALL_MET,
        )
        result = lane.evaluate(*context)
        assert result.max_contribution == 5
        assert result.contribution <= 5

    def test_it_never_produces_a_legal_finding(self, context):
        """The relevant statute requires a conviction with final legal force.
        Reporting on an investigation does not meet that standard at any
        confidence level."""
        lane = ReputationLane(
            index(
                coverage(stage="adjudicated"),
                coverage(source="Tempo", stage="adjudicated", url="https://example.org/b"),
            ),
            ALL_MET,
        )
        result = lane.evaluate(*context)
        assert result.lane is Lane.REPUTATION
        # A lane result carries no statute, article, or severity by
        # construction — those live only on rule findings.
        assert not hasattr(result, "statute")

    def test_allegations_weigh_less_than_adjudicated_outcomes(self, context):
        allegations = ReputationLane(
            index(coverage(), coverage(source="Tempo", url="https://example.org/b")),
            ALL_MET,
        ).evaluate(*context)
        adjudicated = ReputationLane(
            index(
                coverage(stage="adjudicated"),
                coverage(source="Tempo", stage="adjudicated", url="https://example.org/b"),
            ),
            ALL_MET,
        ).evaluate(*context)
        assert adjudicated.contribution > allegations.contribution
