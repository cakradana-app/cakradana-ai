"""Recording that an analyst read a reason code's wording.

What is asserted here is mostly refusal. A decision with nobody's name on it, a
rejection that does not say what was wrong, an "unreviewed" written down as
though it were a decision — each of these produces a ledger that looks like
evidence of a review and is not, and each is refused where it is constructed
rather than discovered later by whoever quotes the coverage figure.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from cakradana.scoring.catalogue import codes
from cakradana.scoring.result import ReviewStatus
from cakradana.scoring.review import (
    REVIEW_FILE,
    ReviewDecision,
    ReviewLedger,
    default_ledger,
)

WHEN = datetime(2026, 8, 17, 9, 0, tzinfo=timezone.utc)
CODE = "FAN_IN_BURST"


def decision(**overrides) -> ReviewDecision:
    defaults = {
        "code": CODE,
        "status": ReviewStatus.VALIDATED,
        "reviewer": "analis@example.org",
        "reviewed_at": WHEN,
        "note": "reads as an observation an analyst can check against the ledger",
    }
    return ReviewDecision(**{**defaults, **overrides})


class TestADecisionNamesSomebody:
    def test_an_acceptance_with_no_reviewer_is_refused(self):
        with pytest.raises(ValidationError, match="names no reviewer"):
            decision(reviewer="")

    def test_whitespace_is_not_a_reviewer(self):
        with pytest.raises(ValidationError, match="names no reviewer"):
            decision(reviewer="   ")

    def test_a_rejection_with_no_reviewer_is_refused(self):
        with pytest.raises(ValidationError, match="names no reviewer"):
            decision(status=ReviewStatus.REJECTED, reviewer="")

    def test_a_decision_must_say_what_was_checked_or_what_was_wrong(self):
        with pytest.raises(ValidationError, match="records no note"):
            decision(note="")

    def test_unreviewed_cannot_be_written_down_as_a_decision(self):
        """It is the absence of one. Recording it would let a code report as
        deliberately left alone, which is not a state anybody decided."""
        with pytest.raises(ValidationError, match="absence of a decision"):
            decision(status=ReviewStatus.UNREVIEWED)

    def test_a_timestamp_with_no_timezone_is_refused(self):
        with pytest.raises(ValidationError, match="no timezone"):
            decision(reviewed_at=datetime(2026, 8, 17, 9, 0))


class TestTheLedgerReportsWhatItHolds:
    def test_a_code_nobody_reviewed_reads_unreviewed(self):
        assert ReviewLedger().status_of(CODE) is ReviewStatus.UNREVIEWED

    def test_a_code_that_does_not_exist_reads_unreviewed(self):
        assert ReviewLedger().status_of("NO_SUCH_CODE") is ReviewStatus.UNREVIEWED

    def test_the_latest_decision_governs(self):
        ledger = ReviewLedger([decision()]).record(
            decision(
                status=ReviewStatus.REJECTED,
                reviewed_at=WHEN + timedelta(days=1),
                note="the comparison figure is a placeholder, not a measured one",
            )
        )
        assert ledger.status_of(CODE) is ReviewStatus.REJECTED

    def test_a_superseded_decision_is_kept(self):
        """A score issued between the two was reported under the first one."""
        ledger = ReviewLedger([decision()]).record(
            decision(
                status=ReviewStatus.REJECTED,
                reviewed_at=WHEN + timedelta(days=1),
                note="the comparison figure is a placeholder",
            )
        )
        assert len(ledger.decisions) == 2
        assert ledger.decisions[0].status is ReviewStatus.VALIDATED

    def test_recording_leaves_the_ledger_it_was_taken_from_alone(self):
        held = ReviewLedger()
        held.record(decision())
        assert held.status_of(CODE) is ReviewStatus.UNREVIEWED


class TestCoverage:
    def test_nothing_reviewed_is_not_complete(self):
        coverage = ReviewLedger().coverage(["A", "B"])
        assert coverage.complete is False
        assert coverage.unreviewed == ("A", "B")

    def test_a_rejected_code_still_being_emitted_is_not_complete(self):
        ledger = ReviewLedger(
            [
                decision(code="A"),
                decision(code="B", status=ReviewStatus.REJECTED, note="misleading"),
            ]
        )
        coverage = ledger.coverage(["A", "B"])
        assert coverage.complete is False
        assert coverage.rejected == ("B",)

    def test_every_code_accepted_is_complete(self):
        ledger = ReviewLedger([decision(code="A"), decision(code="B")])
        assert ledger.coverage(["A", "B"]).complete is True

    def test_an_empty_catalogue_is_not_a_reviewed_one(self):
        """Nothing to measure is not a pass. A check that enumerated nothing
        would otherwise certify a system with no reasons in it."""
        assert ReviewLedger().coverage([]).complete is None

    def test_a_decision_about_a_code_no_longer_emitted_does_not_count(self):
        ledger = ReviewLedger([decision(code="RETIRED_CODE")])
        coverage = ledger.coverage(["A"])
        assert coverage.validated == ()
        assert coverage.stale == ("RETIRED_CODE",)
        assert coverage.complete is False

    def test_the_description_names_what_is_outstanding(self):
        described = ReviewLedger().coverage(["A", "B"]).describe()
        assert "0 of 2" in described
        assert "never read" in described


class TestPersistence:
    def test_a_decision_survives_a_round_trip(self, tmp_path):
        path = tmp_path / "reviews.yaml"
        ReviewLedger([decision()]).save(path)
        loaded = ReviewLedger.load(path)
        assert loaded.status_of(CODE) is ReviewStatus.VALIDATED
        assert loaded.decision_for(CODE).reviewer == "analis@example.org"
        assert loaded.decision_for(CODE).reviewed_at == WHEN

    def test_the_file_is_readable_by_a_person(self, tmp_path):
        path = tmp_path / "reviews.yaml"
        ReviewLedger([decision()]).save(path)
        text = path.read_text(encoding="utf-8")
        assert "analis@example.org" in text
        assert text.startswith("#")

    def test_a_missing_ledger_is_not_an_empty_one(self, tmp_path):
        """'Nobody reviewed anything' and 'the record is not here' are
        different, and only the second is a reason to stop."""
        assert ReviewLedger.load(tmp_path / "absent.yaml") is None

    def test_a_stored_decision_with_no_reviewer_is_refused_on_load(self, tmp_path):
        path = tmp_path / "reviews.yaml"
        path.write_text(
            "decisions:\n"
            "  - code: FAN_IN_BURST\n"
            "    status: validated\n"
            "    reviewer: ''\n"
            "    reviewed_at: '2026-08-17T09:00:00+00:00'\n"
            "    note: accepted\n",
            encoding="utf-8",
        )
        with pytest.raises(ValidationError, match="names no reviewer"):
            ReviewLedger.load(path)


class TestWhatIsShipped:
    def test_the_ledger_ships_with_the_code(self):
        assert REVIEW_FILE.is_file()

    def test_no_reason_code_has_been_reviewed_by_anybody(self):
        """The honest state of this system. It is asserted rather than left
        implicit so that the first recorded review has to be a deliberate
        change to this test as well as to the ledger."""
        coverage = default_ledger().coverage()
        assert coverage.total == len(codes())
        assert coverage.validated == ()
        assert coverage.rejected == ()
        assert len(coverage.unreviewed) == coverage.total
        assert coverage.complete is False
