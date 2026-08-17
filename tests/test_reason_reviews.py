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

from cakradana.cli import main
from cakradana.scoring.catalogue import catalogue, entry_for, stateable_codes
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
    code = overrides.get("code", CODE)
    entry = entry_for(code)
    defaults = {
        "code": code,
        "status": ReviewStatus.VALIDATED,
        "reviewer": "analis@example.org",
        "reviewed_at": WHEN,
        "note": "reads as an observation an analyst can check against the ledger",
        "statements": entry.statements if entry else ("some wording",),
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
            "    note: accepted\n"
            "    statements: ['some wording']\n",
            encoding="utf-8",
        )
        with pytest.raises(ValidationError, match="names no reviewer"):
            ReviewLedger.load(path)


class TestWhatIsShipped:
    def test_the_ledger_ships_with_the_code(self):
        assert REVIEW_FILE.is_file()

    #: The five whose wording a reviewer found misleading. Named rather than
    #: counted, so that removing one from the ledger fails here instead of
    #: quietly reducing the number of sentences this system is answerable for.
    REJECTED = (
        'LAYERING_CHAIN',
        'MODEL_SCORE',
        'PASS_THROUGH',
        'STD_DONASI_SENDER',
        'UNUSUAL_COMBINATION',
    )

    def test_the_shipped_ledger_is_reviewed_but_not_complete(self):
        """The honest state of this system.

        It was every code unreviewed; it is now every code read, with five
        wordings recorded as misleading and still emitted. Both are states the
        gate refuses, and for the same reason: a sentence an analyst acts on
        that nobody can stand behind. The difference is that the remedy is now
        known — rewrite those five — rather than unknown.

        Asserted rather than left implicit so that changing what this system
        has reviewed has to be a deliberate change to this test as well as to
        the ledger.
        """
        coverage = default_ledger().coverage()
        assert coverage.total == len(stateable_codes())
        assert tuple(sorted(coverage.rejected)) == self.REJECTED
        assert len(coverage.validated) == coverage.total - len(self.REJECTED)
        assert coverage.unreviewed == ()
        # Still false, and this is the point. Reviewing a sentence and finding
        # it misleading does not make it fit to ship; it records why it is not.
        assert coverage.complete is False

    def test_a_code_carrying_no_wording_is_not_counted_as_outstanding(self):
        """There is nothing to accept about a sentence that does not exist.

        Counting them would let the figure that gates promotion be raised by
        decisions about wording nobody will ever read."""
        barred = {e.code for e in catalogue() if not e.analyst_facing}
        assert barred
        assert not barred & set(default_ledger().coverage().unreviewed)


class TestTheCommandAnAnalystRuns:
    """`cakradana reason-codes …`.

    The mechanism is only worth having if a person can actually use it, so what
    is asserted here is the round trip: read the wording, record a decision on
    it, and see the coverage move.
    """

    def test_listing_shows_what_nobody_has_read(self, capsys, tmp_path):
        assert main(["reason-codes", "list", "--file", str(tmp_path / "r.yaml")]) == 0
        out = capsys.readouterr().out
        assert "FAN_IN_BURST" in out
        assert "unreviewed" in out

    def test_showing_a_code_prints_the_wording_under_review(self, capsys, tmp_path):
        assert (
            main(
                [
                    "reason-codes",
                    "show",
                    "FAN_IN_BURST",
                    "--file",
                    str(tmp_path / "r.yaml"),
                ]
            )
            == 0
        )
        out = capsys.readouterr().out
        assert "distinct senders" in out
        assert "RULE-T2-01" in out

    def test_showing_a_code_that_does_not_exist_fails(self, capsys, tmp_path):
        assert (
            main(
                ["reason-codes", "show", "NOPE", "--file", str(tmp_path / "r.yaml")]
            )
            == 1
        )

    def test_accepting_records_the_reviewer_and_persists_it(self, capsys, tmp_path):
        path = tmp_path / "r.yaml"
        assert (
            main(
                [
                    "reason-codes",
                    "accept",
                    "FAN_IN_BURST",
                    "--reviewer",
                    "analis@example.org",
                    "--note",
                    "reads as an observation",
                    "--file",
                    str(path),
                ]
            )
            == 0
        )
        capsys.readouterr()
        stored = ReviewLedger.load(path)
        assert stored.status_of("FAN_IN_BURST") is ReviewStatus.VALIDATED
        assert stored.decision_for("FAN_IN_BURST").reviewer == "analis@example.org"

    def test_rejecting_records_the_defect(self, tmp_path, capsys):
        path = tmp_path / "r.yaml"
        main(
            [
                "reason-codes",
                "reject",
                "ADVERSE_COVERAGE",
                "--reviewer",
                "analis@example.org",
                "--note",
                "a reader stops at the first clause and takes it as a finding",
                "--file",
                str(path),
            ]
        )
        capsys.readouterr()
        decision = ReviewLedger.load(path).decision_for("ADVERSE_COVERAGE")
        assert decision.status is ReviewStatus.REJECTED
        assert "first clause" in decision.note

    def test_a_decision_about_a_code_that_is_never_emitted_is_refused(
        self, tmp_path, capsys
    ):
        path = tmp_path / "r.yaml"
        assert (
            main(
                [
                    "reason-codes",
                    "accept",
                    "INVENTED_CODE",
                    "--reviewer",
                    "analis@example.org",
                    "--note",
                    "looks fine",
                    "--file",
                    str(path),
                ]
            )
            == 1
        )
        assert "not a code this system emits" in capsys.readouterr().err
        assert not path.exists()

    def test_a_decision_with_an_empty_note_is_refused(self, tmp_path, capsys):
        assert (
            main(
                [
                    "reason-codes",
                    "accept",
                    "FAN_IN_BURST",
                    "--reviewer",
                    "analis@example.org",
                    "--note",
                    "  ",
                    "--file",
                    str(tmp_path / "r.yaml"),
                ]
            )
            == 1
        )
        assert "records no note" in capsys.readouterr().err

    def test_a_decision_needs_a_reviewer_on_the_command_line(self, tmp_path):
        """Not prompted for, so it cannot be got past by pressing return."""
        with pytest.raises(SystemExit):
            main(
                [
                    "reason-codes",
                    "accept",
                    "FAN_IN_BURST",
                    "--note",
                    "reads as an observation",
                    "--file",
                    str(tmp_path / "r.yaml"),
                ]
            )

    def test_coverage_is_non_zero_while_anything_is_unread(self, tmp_path, capsys):
        assert (
            main(["reason-codes", "coverage", "--file", str(tmp_path / "r.yaml")]) == 1
        )
        assert "0 of" in capsys.readouterr().out

    def test_coverage_moves_when_a_decision_is_recorded(self, tmp_path, capsys):
        path = tmp_path / "r.yaml"
        main(
            [
                "reason-codes",
                "accept",
                "FAN_IN_BURST",
                "--reviewer",
                "analis@example.org",
                "--note",
                "reads as an observation",
                "--file",
                str(path),
            ]
        )
        capsys.readouterr()
        main(["reason-codes", "coverage", "--file", str(path)])
        assert "1 of" in capsys.readouterr().out


class TestAReviewIsOfASentence:
    """An acceptance does not survive the wording it accepted being changed.

    A reviewer reads a sentence, not a code. Carrying their decision across to
    a sentence they never saw would report a review of the current wording that
    nobody performed — which is the failure the ledger exists to prevent, with
    the reviewer's own name attached to it.
    """

    def test_a_decision_records_the_wording_that_was_read(self):
        assert decision().statements == entry_for(CODE).statements

    def test_a_decision_that_records_no_wording_is_refused(self):
        with pytest.raises(ValidationError, match="records no wording"):
            decision(statements=())

    def test_amending_the_wording_puts_the_code_back_in_the_queue(self):
        ledger = ReviewLedger([decision(statements=("an older sentence.",))])
        assert ledger.status_of(CODE) is ReviewStatus.UNREVIEWED

    def test_the_unchanged_wording_keeps_its_decision(self):
        assert ReviewLedger([decision()]).status_of(CODE) is ReviewStatus.VALIDATED

    def test_a_superseded_wording_is_named_apart_from_one_never_read(self):
        """The work needed differs: somebody has to look at what changed,
        not at the code for the first time."""
        ledger = ReviewLedger([decision(statements=("an older sentence.",))])
        coverage = ledger.coverage([CODE, "AMOUNT"])
        assert coverage.superseded == (CODE,)
        assert CODE in coverage.unreviewed
        assert "since changed" in coverage.describe()
        assert coverage.complete is False

    def test_a_rejection_of_wording_since_amended_does_not_block_forever(self):
        """A rejected sentence that was then rewritten is a different sentence,
        and blocking on the old finding would make the fix invisible."""
        ledger = ReviewLedger(
            [
                decision(
                    status=ReviewStatus.REJECTED,
                    statements=("an older sentence.",),
                    note="the comparison figure is a placeholder",
                )
            ]
        )
        assert ledger.status_of(CODE) is ReviewStatus.UNREVIEWED
        assert ledger.coverage([CODE]).rejected == ()

    def test_the_recorded_wording_survives_a_round_trip(self, tmp_path):
        path = tmp_path / "r.yaml"
        ReviewLedger([decision()]).save(path)
        assert ReviewLedger.load(path).decision_for(CODE).statements == (
            entry_for(CODE).statements
        )
        assert ReviewLedger.load(path).status_of(CODE) is ReviewStatus.VALIDATED

    def test_the_command_records_the_wording_it_showed(self, tmp_path, capsys):
        path = tmp_path / "r.yaml"
        main(
            [
                "reason-codes",
                "accept",
                CODE,
                "--reviewer",
                "analis@example.org",
                "--note",
                "reads as an observation",
                "--file",
                str(path),
            ]
        )
        capsys.readouterr()
        stored = ReviewLedger.load(path)
        assert stored.decision_for(CODE).statements == entry_for(CODE).statements
        assert stored.status_of(CODE) is ReviewStatus.VALIDATED
