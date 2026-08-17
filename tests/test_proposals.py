"""Drafted dispositions, and the line between a reading and a decision.

The review ledger sat empty with the mechanism complete and the gate correctly
blocking, because reading fifty-one sentences from a blank page is work nobody
schedules. A draft turns that into agreeing or disagreeing.

The risk a draft introduces is the one the review module was built to refuse: a
way to accept fifty sentences without reading them, producing a ledger that
certifies nothing while looking complete. So the tests that matter here are the
ones establishing that a proposal cannot become a decision by any path.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cakradana.scoring.catalogue import catalogue, entry_for
from cakradana.scoring.proposals import (
    PROPOSAL_FILE,
    ProposalsInvalid,
    _load,
    drafted,
    proposal_for,
)
from cakradana.scoring.review import REVIEW_FILE, ReviewLedger
from cakradana.scoring.result import ReviewStatus


def _reload(path: Path):
    _load.cache_clear()
    return _load(str(path))


def test_every_analyst_facing_code_has_a_reading():
    # The point of the worksheet is that nobody meets a blank sentence. A code
    # with no proposal is one the reviewer has to form a view on unaided, which
    # is the state this exists to remove.
    for entry in catalogue():
        if not entry.analyst_facing:
            continue
        assert proposal_for(entry.code) is not None, entry.code


def test_a_reading_is_not_a_decision():
    # The property everything else here rests on. Reading every proposal must
    # leave the ledger exactly as it was, whatever state that is.
    #
    # This used to be checked by asserting nothing was validated, which was a
    # sound proxy only while the ledger was empty. Once decisions were recorded
    # it would have failed against a working system — and, worse, a version of
    # it weakened to keep passing would have stopped testing anything. So the
    # invariant is now stated directly: every status is the same afterwards as
    # before, and reading is what changed nothing.
    ledger = ReviewLedger.load(REVIEW_FILE) or ReviewLedger()
    before = ledger.coverage()
    statuses_before = {
        entry.code: ledger.status_of(entry.code)
        for entry in catalogue()
        if entry.analyst_facing
    }

    for entry in catalogue():
        if entry.analyst_facing:
            proposal_for(entry.code)

    reloaded = ReviewLedger.load(REVIEW_FILE) or ReviewLedger()
    assert reloaded.coverage() == before
    assert {
        entry.code: reloaded.status_of(entry.code)
        for entry in catalogue()
        if entry.analyst_facing
    } == statuses_before


def test_a_proposal_carries_its_reasoning():
    # A proposal with no reasoning is a vote, and a reviewer cannot agree or
    # disagree with a vote — they can only copy it, which is the outcome that
    # makes a ledger worthless.
    for entry in catalogue():
        if not entry.analyst_facing:
            continue
        proposal = proposal_for(entry.code)
        assert proposal.note.strip()
        assert proposal.propose in {"accept", "reject"}


def test_grouped_and_individual_readings_are_distinguishable():
    # A reviewer is entitled to know whether the paragraph in front of them was
    # written about this sentence or about a class it belongs to. They carry
    # different weight and the worksheet says which.
    individual = {proposal.code for proposal in drafted()}
    assert individual, "no individual reading was written for anything"
    assert proposal_for(next(iter(individual))).grouped is False

    ungrouped = [
        entry.code
        for entry in catalogue()
        if entry.analyst_facing and entry.code not in individual
    ]
    assert ungrouped, "every code was written about individually; the default is dead"
    assert proposal_for(ungrouped[0]).grouped is True


def test_the_reading_disagrees_with_something():
    # A draft that accepts everything is not a reading, it is a rubber stamp
    # wearing the shape of one — and it is the failure mode a reviewer starting
    # from a draft is least likely to catch, because agreeing is the cheap path.
    rejected = [proposal for proposal in drafted() if proposal.propose == "reject"]
    assert rejected, "the draft accepts every code, which is not a review"


def test_a_proposal_about_an_unknown_code_is_refused(tmp_path):
    # The shape a stale file takes after a code is renamed. It fails rather than
    # being skipped, because a silently-dropped reading is a code the worksheet
    # presents as unread while somebody believes it was covered.
    path = tmp_path / "proposals.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "proposals": [
                    {"code": "NO_SUCH_CODE", "propose": "accept", "note": "x"}
                ],
            }
        )
    )
    with pytest.raises(ProposalsInvalid, match="not a code this system emits"):
        _reload(path)


def test_a_proposal_about_a_code_with_no_wording_is_refused(tmp_path):
    barred = next(
        (entry.code for entry in catalogue() if not entry.analyst_facing), None
    )
    if barred is None:
        pytest.skip("every code in this catalogue carries wording")
    path = tmp_path / "proposals.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "proposals": [{"code": barred, "propose": "accept", "note": "x"}],
            }
        )
    )
    with pytest.raises(ProposalsInvalid, match="nothing to propose"):
        _reload(path)


def test_a_proposal_without_reasoning_is_refused(tmp_path):
    code = next(entry.code for entry in catalogue() if entry.analyst_facing)
    path = tmp_path / "proposals.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "proposals": [{"code": code, "propose": "accept", "note": "   "}],
            }
        )
    )
    with pytest.raises(ProposalsInvalid, match="a vote"):
        _reload(path)


def test_a_code_proposed_on_twice_is_refused(tmp_path):
    code = next(entry.code for entry in catalogue() if entry.analyst_facing)
    path = tmp_path / "proposals.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "proposals": [
                    {"code": code, "propose": "accept", "note": "one"},
                    {"code": code, "propose": "reject", "note": "two"},
                ],
            }
        )
    )
    with pytest.raises(ProposalsInvalid, match="proposed on twice"):
        _reload(path)


def test_the_shipped_file_is_valid():
    # Run last so the cache is left holding the real file rather than a fixture.
    _reload(PROPOSAL_FILE)
    assert drafted()


def test_every_individual_reading_names_a_real_sentence():
    # A reading that quotes wording the catalogue no longer carries describes a
    # sentence nobody will be shown.
    for proposal in drafted():
        assert entry_for(proposal.code) is not None
        assert entry_for(proposal.code).analyst_facing
