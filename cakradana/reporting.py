"""Campaign finance submissions, for reconciliation.

Recipients file what they received. Reconciling those filings against the
donations the system knows about detects an offence currently invisible to
everyone: a donation that reached a campaign and never appeared in its report.

The comparison is only meaningful when the filing is complete for the period
being compared. A submission covering January against donations from March
would report every March donation as unreported, which would be an accusation
manufactured out of a date range. So a set declares which periods it covers,
and a donation outside those periods yields indeterminate rather than a finding.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Iterable

from pydantic import BaseModel, ConfigDict


class ReportedDonation(BaseModel):
    """One line of a filed campaign finance report."""

    model_config = ConfigDict(frozen=True)

    electoral_context: str
    #: The report this line came from — the initial statement, the periodic
    #: receipts return, or the final account.
    report_kind: str
    donor_name: str | None = None
    donor_ref: str | None = None
    recipient_ref: str | None = None
    amount_idr: int | None = None
    occurred_on: date | None = None


class SubmissionSet:
    """Filed reports, and the periods they are complete for."""

    def __init__(
        self,
        lines: Iterable[ReportedDonation] = (),
        *,
        covered_periods: Iterable[tuple[str, date, date]] = (),
        available: bool = False,
        authoritative: bool = True,
        unavailable_reason: str = (
            "campaign finance submissions have not been made available for "
            "reconciliation"
        ),
    ) -> None:
        self.available = available
        self.authoritative = authoritative
        self.unavailable_reason = unavailable_reason
        self._lines = tuple(lines)
        self._periods = tuple(covered_periods)

    def __len__(self) -> int:
        return len(self._lines)

    def covers(self, electoral_context: str | None, when: date) -> bool:
        if electoral_context is None:
            return False
        return any(
            context == electoral_context and start <= when <= end
            for context, start, end in self._periods
        )

    def contains(
        self,
        *,
        electoral_context: str | None,
        donor_ref: str | None,
        recipient_ref: str | None,
        amount_idr: int,
        occurred_on: date,
        amount_tolerance: float = 0.02,
        day_tolerance: int = 3,
    ) -> bool:
        """Whether a filing plausibly records this donation.

        Matching is tolerant on both amount and date. A report transcribed by
        hand will round a figure and misplace a day, and treating either as a
        mismatch would report a donation that was properly declared as
        undeclared — the most damaging error this rule can make.
        """
        for line in self._lines:
            if line.electoral_context != electoral_context:
                continue
            if donor_ref and line.donor_ref and line.donor_ref != donor_ref:
                continue
            if recipient_ref and line.recipient_ref and line.recipient_ref != recipient_ref:
                continue
            if line.amount_idr is not None:
                spread = abs(line.amount_idr - amount_idr) / max(amount_idr, 1)
                if spread > amount_tolerance:
                    continue
            if line.occurred_on is not None:
                if abs((line.occurred_on - occurred_on).days) > day_tolerance:
                    continue
            return True
        return False


def no_submissions() -> SubmissionSet:
    """The default: nothing filed has been shared with this system."""
    return SubmissionSet()
