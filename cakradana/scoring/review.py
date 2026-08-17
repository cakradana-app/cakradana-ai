"""Whether an analyst has read a reason code's wording, and who.

The catalogue says what the system can say. This says whether anybody has read
it. They are separate because they change for different reasons and on
different timescales: the wording changes when the code changes, the review
changes when a person sits down and reads it.

A decision names the person who made it. A record that a code was accepted, with
nobody's name on it, is not evidence that anybody accepted it — it is a claim
about a review that may never have happened, and it is refused at construction
rather than stored and discovered later. The same applies to a rejection: a code
found misleading has to say what was misleading about it, or the finding cannot
be acted on.

There is no bulk-accept. Reviewing wording means reading the sentence, and a
mechanism for accepting fifty sentences without reading them would produce a
ledger that certifies exactly nothing while looking complete.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from cakradana.scoring.catalogue import ReviewStatus, codes

#: Reviews live beside the wording they are about, so that a change to a
#: statement and the review of that statement land in the same history and a
#: reviewer reading a diff can see both.
REVIEW_FILE = Path(__file__).resolve().parent / "reason_code_reviews.yaml"

LEDGER_VERSION = 1


class ReviewRefused(ValueError):
    """Raised when a decision is about something that cannot be reviewed."""


class ReviewDecision(BaseModel):
    """One analyst's decision about one code's wording."""

    model_config = ConfigDict(frozen=True)

    code: str
    status: ReviewStatus
    #: Who read it. Not optional, and not defaulted to the current user: the
    #: value of the review is that a named person is answerable for it.
    reviewer: str
    reviewed_at: datetime
    #: What they concluded. For a rejection this is the defect; for an
    #: acceptance it is what they checked.
    note: str

    @model_validator(mode="after")
    def _accountable(self) -> ReviewDecision:
        if self.status is ReviewStatus.UNREVIEWED:
            raise ValueError(
                "unreviewed is the absence of a decision and cannot be recorded "
                "as one; delete the decision instead"
            )
        if not self.reviewer.strip():
            raise ValueError(
                f"the review of {self.code} names no reviewer; a decision "
                f"nobody is answerable for is not a review"
            )
        if not self.note.strip():
            raise ValueError(
                f"the review of {self.code} records no note; an acceptance has "
                f"to say what was checked and a rejection has to say what was "
                f"wrong, or neither can be acted on"
            )
        if self.reviewed_at.tzinfo is None:
            raise ValueError(
                f"the review of {self.code} carries a timestamp with no "
                f"timezone; when a decision was taken has to be unambiguous"
            )
        return self


@dataclass(frozen=True)
class ReviewCoverage:
    """How much of the catalogue anybody has actually read."""

    total: int
    validated: tuple[str, ...]
    rejected: tuple[str, ...]
    unreviewed: tuple[str, ...]
    #: Decisions about codes the catalogue no longer declares. Reported rather
    #: than dropped: a review of wording that no longer exists is not coverage,
    #: and silently counting it would inflate the figure that gates promotion.
    stale: tuple[str, ...] = ()

    @property
    def complete(self) -> bool | None:
        """Whether every code has been read and accepted.

        None when there is nothing to measure. An empty catalogue is not a
        fully reviewed one, and reporting it as complete would let the check
        pass by failing to enumerate anything.
        """
        if not self.total:
            return None
        return not self.rejected and not self.unreviewed

    def describe(self) -> str:
        parts = [
            f"{len(self.validated)} of {self.total} reason codes reviewed and "
            f"accepted"
        ]
        if self.rejected:
            parts.append(
                f"{len(self.rejected)} found misleading and still emitted "
                f"({', '.join(self.rejected)})"
            )
        if self.unreviewed:
            shown = ", ".join(self.unreviewed[:5])
            more = "" if len(self.unreviewed) <= 5 else ", …"
            parts.append(f"{len(self.unreviewed)} never read ({shown}{more})")
        if self.stale:
            parts.append(
                f"{len(self.stale)} decision(s) about codes no longer emitted "
                f"({', '.join(self.stale)})"
            )
        return "; ".join(parts)


class ReviewLedger:
    """The decisions taken so far, latest per code.

    Immutable. ``record`` returns a new ledger rather than mutating this one,
    so a caller holding a ledger holds the state it read and not whatever
    happened to it since.
    """

    def __init__(self, decisions: Iterable[ReviewDecision] = ()) -> None:
        self._decisions: tuple[ReviewDecision, ...] = tuple(decisions)
        latest: dict[str, ReviewDecision] = {}
        for decision in self._decisions:
            held = latest.get(decision.code)
            if held is None or decision.reviewed_at >= held.reviewed_at:
                latest[decision.code] = decision
        self._latest = latest

    def __len__(self) -> int:
        return len(self._decisions)

    @property
    def decisions(self) -> tuple[ReviewDecision, ...]:
        """Every decision ever recorded, in the order it was recorded.

        Superseded decisions are kept. An analyst who accepted a wording and
        later rejected it leaves two facts behind, and the first one is what a
        score issued in between was reported under.
        """
        return self._decisions

    def decision_for(self, code: str) -> ReviewDecision | None:
        return self._latest.get(code)

    def status_of(self, code: str) -> ReviewStatus:
        decision = self._latest.get(code)
        return decision.status if decision else ReviewStatus.UNREVIEWED

    def statuses(self, over: Sequence[str] | None = None) -> dict[str, ReviewStatus]:
        declared = tuple(over) if over is not None else codes()
        return {code: self.status_of(code) for code in declared}

    def record(self, decision: ReviewDecision) -> ReviewLedger:
        return ReviewLedger(self._decisions + (decision,))

    def coverage(self, over: Sequence[str] | None = None) -> ReviewCoverage:
        declared = tuple(over) if over is not None else codes()
        statuses = self.statuses(declared)
        return ReviewCoverage(
            total=len(declared),
            validated=tuple(
                c for c in declared if statuses[c] is ReviewStatus.VALIDATED
            ),
            rejected=tuple(
                c for c in declared if statuses[c] is ReviewStatus.REJECTED
            ),
            unreviewed=tuple(
                c for c in declared if statuses[c] is ReviewStatus.UNREVIEWED
            ),
            stale=tuple(sorted(set(self._latest) - set(declared))),
        )

    # -- persistence ------------------------------------------------------

    @classmethod
    def load(cls, path: Path = REVIEW_FILE) -> ReviewLedger | None:
        """Read the ledger, or None when there is no ledger to read.

        A missing file is not an empty ledger. "Nobody has reviewed anything"
        and "the record of who reviewed what is not here" are different
        statements, and the second one is a reason to stop rather than a figure
        to report.
        """
        if not path.is_file():
            return None
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls(
            ReviewDecision(**entry) for entry in (loaded.get("decisions") or ())
        )

    def save(self, path: Path = REVIEW_FILE) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": LEDGER_VERSION,
            "decisions": [
                {
                    "code": d.code,
                    "status": d.status.value,
                    "reviewer": d.reviewer,
                    "reviewed_at": d.reviewed_at.isoformat(),
                    "note": d.note,
                }
                for d in self._decisions
            ],
        }
        path.write_text(
            _LEDGER_HEADER
            + yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )


_LEDGER_HEADER = """\
# Analyst review of reason-code wording.
#
# One entry per decision, kept rather than overwritten: a code accepted and
# later rejected leaves both facts behind, and the first is what any score
# issued in between was reported under.
#
# Written by `cakradana reason-codes accept|reject`. Edit by hand only to
# correct a transcription error, and commit the change like any other.
"""


@lru_cache(maxsize=1)
def default_ledger() -> ReviewLedger | None:
    """The ledger shipped with the code, cached for the life of the process.

    Cached because scoring consults it per reason and the file changes only
    when somebody records a decision. ``default_ledger.cache_clear()`` after
    writing.
    """
    return ReviewLedger.load()


def default_statuses() -> dict[str, ReviewStatus]:
    """Every catalogued code's review state, for stamping onto a result.

    With no ledger present every code reads unreviewed. That is the safe
    direction: an absent record of review cannot make a wording look accepted.
    """
    ledger = default_ledger()
    if ledger is None:
        return {code: ReviewStatus.UNREVIEWED for code in codes()}
    return ledger.statuses()


def now() -> datetime:
    return datetime.now(tz=timezone.utc)
