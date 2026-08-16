"""Score composition.

What is asserted here is mostly separation: that a statutory finding and a
behavioural estimate never become one number, that a partly evaluated donation
never reads as a clean one, and that a score assembled from half the lanes
never passes for a complete one.
"""

from __future__ import annotations

from datetime import date

import pytest

from cakradana.calendar import CampaignPeriod, ElectoralCalendar
from cakradana.history import InMemoryDonationStore
from cakradana.rules import load_latest
from cakradana.scoring import (
    Band,
    Lane,
    MissingReasons,
    Reason,
    ScoreComposer,
    band_for,
    contribution_from,
    unavailable,
)
from cakradana.scoring.scorer import Scorer
from cakradana.schema import EntityType
from tests.conftest import at, make_donation

CONTEXT = "pemilu-2029"
INDIVIDUAL_PARTY_LIMIT = 200_000_000


@pytest.fixture(scope="module")
def calendar():
    return ElectoralCalendar(
        [
            CampaignPeriod(
                electoral_context=CONTEXT,
                start=date(2026, 9, 1),
                end=date(2026, 11, 24),
                reporting_deadlines=(date(2026, 10, 15),),
            )
        ]
    )


@pytest.fixture
def scorer(calendar):
    return Scorer(load_latest(), calendar=calendar, require_verified_citations=False)


def donation(**kwargs):
    kwargs.setdefault("electoral_context", CONTEXT)
    kwargs.setdefault("occurred", at(2026, 3, 1))
    return make_donation(**kwargs)


def score(scorer, target, others=()):
    store = InMemoryDonationStore([*others, target])
    result, _ = scorer.score(target, store.knowable_at(target.occurred_at))
    return result


class TestSeparation:
    def test_crossing_a_statutory_limit_does_not_move_the_behavioural_score(
        self, scorer
    ):
        """Averaging a statutory fact with an estimate yields something that is
        neither auditable as a finding nor readable as a probability.

        Two donations identical in structure, one lawful and one not, must
        receive the same behavioural score. The breach belongs in the findings
        and nowhere else. A donation can legitimately carry both a finding and
        a behavioural signal — what it must not do is let one become the other.
        """
        lawful = score(scorer, donation(amount_idr=INDIVIDUAL_PARTY_LIMIT - 1))
        breaching = score(scorer, donation(amount_idr=INDIVIDUAL_PARTY_LIMIT * 3))

        assert not lawful.legal_findings
        assert breaching.legal_findings
        assert lawful.behavioural.score == breaching.behavioural.score

    def test_legal_findings_carry_a_citation_and_the_observed_value(self, scorer):
        result = score(scorer, donation(amount_idr=INDIVIDUAL_PARTY_LIMIT * 3))
        finding = result.legal_findings[0]
        assert finding.statute and finding.article
        assert finding.threshold_idr == INDIVIDUAL_PARTY_LIMIT
        assert finding.observed

    def test_behavioural_reasons_never_carry_a_statute(self, scorer):
        result = score(scorer, donation(amount_idr=INDIVIDUAL_PARTY_LIMIT * 3))
        for reason in result.behavioural.reasons:
            assert "Pasal" not in reason.statement


class TestPartialEvaluation:
    def test_unevaluated_rules_are_always_reported(self, scorer):
        result = score(scorer, donation())
        assert result.indeterminate_rules
        assert not result.is_fully_evaluated

    def test_absence_of_findings_is_not_a_clean_bill(self, scorer):
        """A donation with no findings and unevaluated prohibitions has been
        partly examined, not cleared."""
        result = score(scorer, donation(amount_idr=1_000_000))
        assert not result.legal_findings
        assert not result.is_fully_evaluated

    def test_the_unavailable_register_is_named(self, scorer):
        result = score(scorer, donation())
        reasons = " ".join(r.reason or "" for r in result.indeterminate_rules)
        assert "register" in reasons


class TestDegradedScoring:
    def test_missing_lanes_lower_the_attainable_maximum(self, scorer):
        """Rescaling the remaining lanes to fill the gap would make a partial
        score indistinguishable from a complete one."""
        result = score(scorer, donation())
        assert result.behavioural.degraded
        assert result.behavioural.attainable_max == 30

    def test_an_absent_lane_is_reported_rather_than_omitted(self, scorer):
        result = score(scorer, donation())
        lanes = {lane.lane: lane for lane in result.behavioural.lanes}
        assert set(lanes) == set(Lane)
        assert lanes[Lane.CLASSIFIER].unavailable_reason

    def test_the_reputation_lane_is_off_by_default(self, scorer):
        """It accuses named parties on the strength of press coverage, so it
        does not run until its controls are in place."""
        result = score(scorer, donation())
        lanes = {lane.lane: lane for lane in result.behavioural.lanes}
        assert not lanes[Lane.REPUTATION].available


class TestComposer:
    def test_a_score_without_reasons_is_withheld(self):
        """A bare number cannot be triaged, contested, or audited."""
        composer = ScoreComposer()
        with pytest.raises(MissingReasons):
            composer._behavioural(
                (contribution_from(Lane.CLASSIFIER, 0.8, ()),)
            )

    def test_lane_contributions_are_capped_at_their_share(self):
        lane = contribution_from(
            Lane.ANOMALY,
            1.0,
            (Reason(code="X", lane=Lane.ANOMALY, weight=0.5, statement="s"),),
        )
        assert lane.contribution == 15

    def test_an_exploratory_lane_cannot_outweigh_the_classifier(self):
        anomaly = contribution_from(
            Lane.ANOMALY, 1.0,
            (Reason(code="A", lane=Lane.ANOMALY, weight=0.5, statement="s"),),
        )
        classifier = contribution_from(
            Lane.CLASSIFIER, 1.0,
            (Reason(code="C", lane=Lane.CLASSIFIER, weight=0.9, statement="s"),),
        )
        assert classifier.contribution > anomaly.contribution * 3

    def test_reasons_are_ordered_by_contribution(self):
        composer = ScoreComposer()
        lane = contribution_from(
            Lane.GRAPH,
            0.5,
            (
                Reason(code="LOW", lane=Lane.GRAPH, weight=0.2, statement="s"),
                Reason(code="HIGH", lane=Lane.GRAPH, weight=0.9, statement="s"),
            ),
        )
        result = composer._behavioural((lane,))
        assert [r.code for r in result.reasons] == ["HIGH", "LOW"]

    def test_unavailable_lanes_become_reasons_too(self):
        composer = ScoreComposer()
        result = composer._behavioural(
            (unavailable(Lane.REPUTATION, "no match"),)
        )
        assert any(r.code == "LANE_UNAVAILABLE" for r in result.reasons)


class TestBands:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (0, Band.LOW),
            (24, Band.LOW),
            (25, Band.MODERATE),
            (49, Band.MODERATE),
            (50, Band.HIGH),
            (74, Band.HIGH),
            (75, Band.CRITICAL),
            (100, Band.CRITICAL),
        ],
    )
    def test_band_boundaries(self, value, expected):
        assert band_for(value) is expected


class TestVersioning:
    def test_every_result_records_the_versions_that_produced_it(self, scorer):
        result = score(scorer, donation())
        assert result.versions.rule_set
        assert result.versions.features.startswith("features-")

    def test_the_model_version_is_absent_when_no_model_ran(self, scorer):
        assert score(scorer, donation()).versions.model is None
