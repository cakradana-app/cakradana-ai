"""Drafted dispositions a reviewer starts from, which are not decisions.

Reviewing fifty-one sentences from a blank page is work nobody schedules, and
the review ledger sat empty for exactly that reason: the mechanism was built,
the gate correctly blocked, and no analyst had a way in that took less than an
afternoon. A draft changes the task from "read fifty-one sentences and form a
view" to "read fifty-one sentences and agree or disagree", which is a task a
person actually completes.

The distinction this module exists to hold is between the two. A proposal is
somebody's reading; a decision is a named person's conclusion, recorded with
what they checked. Nothing here writes to the ledger, nothing here changes a
code's status, and nothing here opens the promotion gate — a draft that could do
any of those would be a bulk accept wearing a different name, and the review
module refuses bulk accepts on the ground that a mechanism for accepting fifty
sentences without reading them produces a ledger certifying nothing while
looking complete.

Codes with no individual proposal fall to the default one, which states what was
checked rather than asserting a conclusion per code. Forty-one separate
paragraphs saying the same thing would read as forty-one readings and be one.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import yaml

from cakradana.scoring.catalogue import catalogue, entry_for

PROPOSAL_FILE = Path(__file__).resolve().parent / "reason_code_proposals.yaml"


class ProposalsInvalid(ValueError):
    """Raised when the drafted proposals do not describe this catalogue."""


@dataclass(frozen=True)
class Proposal:
    """One drafted disposition, and why."""

    code: str
    propose: str
    note: str
    #: What the reading checked. Present on the grouped default, where the
    #: reasoning is about a class rather than about one sentence.
    checked: tuple[str, ...] = ()
    #: Whether this came from the default block rather than being written for
    #: this code. A reviewer is entitled to know which they are reading.
    grouped: bool = False


def _validate(payload: dict) -> None:
    if payload.get("version") != 1:
        raise ProposalsInvalid("proposals file must declare version: 1")

    known = {entry.code for entry in catalogue()}
    seen: set[str] = set()
    for raw in payload.get("proposals", []):
        code = raw.get("code")
        if code in seen:
            raise ProposalsInvalid(f"{code} is proposed on twice")
        seen.add(code)
        # A proposal about a code that does not exist is a reading of something
        # nobody will review, and it is the shape a stale file takes after a
        # code is renamed — so it fails rather than being skipped.
        if code not in known:
            raise ProposalsInvalid(
                f"{code} is proposed on and is not a code this system emits"
            )
        entry = entry_for(code)
        if not entry.analyst_facing:
            raise ProposalsInvalid(
                f"{code} carries no wording and is never shown to anybody, so "
                f"there is nothing to propose a disposition on"
            )
        if raw.get("propose") not in {"accept", "reject"}:
            raise ProposalsInvalid(f"{code}: propose must be accept or reject")
        if not str(raw.get("note", "")).strip():
            raise ProposalsInvalid(
                f"{code}: a proposal with no reasoning is a vote, and a reviewer "
                f"cannot agree or disagree with a vote"
            )

    default = payload.get("default")
    if default is not None:
        if default.get("propose") not in {"accept", "reject"}:
            raise ProposalsInvalid("default: propose must be accept or reject")
        if not str(default.get("note", "")).strip():
            raise ProposalsInvalid("default: a proposal with no reasoning is a vote")


@lru_cache(maxsize=1)
def _load(path: str) -> tuple[dict[str, Proposal], Proposal | None]:
    payload = yaml.safe_load(Path(path).read_text()) or {}
    _validate(payload)

    individual = {
        raw["code"]: Proposal(
            code=raw["code"],
            propose=raw["propose"],
            note=" ".join(str(raw["note"]).split()),
            checked=tuple(raw.get("checked", ())),
        )
        for raw in payload.get("proposals", [])
    }
    raw_default = payload.get("default")
    default = (
        Proposal(
            code="",
            propose=raw_default["propose"],
            note=" ".join(str(raw_default["note"]).split()),
            checked=tuple(raw_default.get("checked", ())),
            grouped=True,
        )
        if raw_default
        else None
    )
    return individual, default


def proposal_for(code: str, path: Path = PROPOSAL_FILE) -> Proposal | None:
    """The drafted disposition for one code, individual or grouped."""
    individual, default = _load(str(path))
    if code in individual:
        return individual[code]
    if default is None:
        return None
    return Proposal(
        code=code,
        propose=default.propose,
        note=default.note,
        checked=default.checked,
        grouped=True,
    )


def drafted(path: Path = PROPOSAL_FILE) -> Sequence[Proposal]:
    """Every individually drafted proposal, in catalogue order."""
    individual, _ = _load(str(path))
    return tuple(
        individual[entry.code] for entry in catalogue() if entry.code in individual
    )
