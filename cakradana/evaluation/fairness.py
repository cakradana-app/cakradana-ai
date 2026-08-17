"""Whether the system treats comparable subjects comparably.

Two questions are asked here, and they are not the same question.

**Does the output track political affiliation?** A system that scores donations
to one party higher than donations to another, for reasons that are not in the
statute, is not a transparency tool. It is a campaign instrument, and it would
be used as one. The defence that no protected attribute is a model input does
not survive contact with the data: affiliation arrives through district, through
donor communities, through which reports are digitised well enough to parse, and
through a label loop that learns from whichever subjects were investigated
first.

**Does performance differ by group?** Precision and recall are averages, and an
average conceals a subgroup for whom the system is much worse. A district whose
donations are wrongly flagged at three times the rate of another is a district
where the system is doing harm, and no aggregate figure would show it.

What is measured, and what is not
--------------------------------

Equal flag rates across parties would be the wrong target. If one party genuinely
receives more over-limit donations, a system that flags them equally is broken in
the other direction. So the raw selection rate is reported as description, never
as a verdict.

The gate is on **error** rates: of the donations a group's subjects had that
review confirmed were fine, what share did the system flag anyway? That figure
has no legitimate reason to differ by party or district. When it does, the
difference is the system's, not the population's.

Everything here refuses to produce a number it cannot support. A group with too
few reviewed donations gets ``None`` and a reason, on the same principle that
governs recall estimation: a fabricated rate is worse than an absent one,
because only the fabricated one gets quoted.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from cakradana.evaluation.sampling import wilson_interval

#: Below this many reviewed donations a group rate is noise. Chosen so the 95%
#: Wilson interval on a mid-range rate is narrower than the disparity threshold
#: being tested; under it the comparison could not distinguish a real gap from
#: sampling variation, so no rate is reported.
MIN_GROUP_REVIEWED = 30

#: Largest defensible ratio between the highest and lowest group false-flag
#: rate. The four-fifths convention, expressed as a maximum rather than a
#: minimum: a group flagged in error more than 1.25 times as often as another
#: is being treated differently, and the difference has to be explained before
#: the model serves.
MAX_FALSE_FLAG_DISPARITY = 1.25

#: Donation size bands, in rupiah. Boundaries follow the statutory limits rather
#: than round numbers, because those are the points where behaviour changes: a
#: donor sizing a contribution against a limit behaves differently from one who
#: is nowhere near it.
SIZE_BANDS: tuple[tuple[str, int, int | None], ...] = (
    ("under_10jt", 0, 10_000_000),
    ("10jt_to_200jt", 10_000_000, 200_000_000),
    ("200jt_to_2_5m", 200_000_000, 2_500_000_000),
    ("over_2_5m", 2_500_000_000, None),
)


def size_band(amount_idr: int) -> str:
    """Which band an amount falls in."""
    for name, lower, upper in SIZE_BANDS:
        if amount_idr >= lower and (upper is None or amount_idr < upper):
            return name
    return SIZE_BANDS[-1][0]


@dataclass(frozen=True)
class Cohort:
    """One scored donation, with the attributes a breakdown is taken over.

    ``affiliation`` is the recipient's party. It is carried here so it can be
    tested against, and it is never a model input; the whole point of the test
    is that keeping it out of the features does not keep it out of the output.

    ``reviewed`` distinguishes a donation a human actually adjudicated from one
    nobody looked at. Only reviewed donations carry information about whether a
    flag was right, so only they enter the error rates.
    """

    donation_id: str
    score: float
    flagged: bool
    #: Recipient party. None when the recipient is unresolved or non-partisan.
    affiliation: str | None = None
    district: str | None = None
    recipient_type: str | None = None
    amount_idr: int = 0
    #: Whether a human adjudicated this donation.
    reviewed: bool = False
    #: The adjudicated outcome. Meaningless unless ``reviewed``.
    confirmed_risky: bool = False

    @property
    def band(self) -> str:
        return size_band(self.amount_idr)


@dataclass(frozen=True)
class GroupPerformance:
    """How the system behaved for one group.

    Every rate is ``None`` rather than zero when its denominator is too small.
    A group of four donations produces a false-flag rate of 0.00 or 0.25 and
    nothing in between, and neither figure means anything.
    """

    group: str
    total: int
    reviewed: int
    flagged: int
    confirmed_risky: int
    #: Of those flagged and reviewed, the share confirmed risky.
    precision: float | None
    #: Of those reviewed and confirmed *not* risky, the share flagged anyway.
    #: The rate the gate is taken on.
    false_flag_rate: float | None
    #: Of those confirmed risky, the share flagged. Recall within the group.
    detection_rate: float | None
    #: 95% interval on the false-flag rate, or None alongside it.
    false_flag_interval: tuple[float, float] | None
    unmeasurable_reason: str | None = None

    @property
    def selection_rate(self) -> float:
        """Share of the group's donations that were flagged.

        Description, not a verdict. Groups legitimately differ here.
        """
        return self.flagged / self.total if self.total else 0.0

    def describe(self) -> str:
        if self.false_flag_rate is None:
            return f"{self.group}: not measurable — {self.unmeasurable_reason}"
        low, high = self.false_flag_interval or (0.0, 1.0)
        return (
            f"{self.group}: false-flag {self.false_flag_rate:.3f} "
            f"[{low:.3f}, {high:.3f}] over {self.reviewed} reviewed "
            f"(selection {self.selection_rate:.3f})"
        )


@dataclass(frozen=True)
class DifferentialReport:
    """Performance broken down by one attribute."""

    attribute: str
    groups: tuple[GroupPerformance, ...]
    #: Ratio of the highest group false-flag rate to the lowest. None when
    #: fewer than two groups could be measured.
    disparity: float | None
    #: The groups the disparity is between, worst first.
    extremes: tuple[str, str] | None
    unmeasurable_reason: str | None = None

    @property
    def measurable(self) -> bool:
        return self.disparity is not None

    @property
    def within_tolerance(self) -> bool | None:
        """Whether the spread is defensible.

        None when it could not be measured — which is not the same as passing,
        and callers must not treat it as passing.
        """
        if self.disparity is None:
            return None
        return self.disparity <= MAX_FALSE_FLAG_DISPARITY

    def concerns(self) -> tuple[str, ...]:
        found: list[str] = []
        if self.disparity is None:
            found.append(
                f"{self.attribute}: no disparity figure — "
                f"{self.unmeasurable_reason}"
            )
            return tuple(found)
        if self.disparity > MAX_FALSE_FLAG_DISPARITY and self.extremes:
            worst, best = self.extremes
            found.append(
                f"{self.attribute}: {worst} is flagged in error "
                f"{self.disparity:.2f}x as often as {best}, above the "
                f"{MAX_FALSE_FLAG_DISPARITY:.2f} tolerance"
            )
        unmeasured = [g.group for g in self.groups if g.false_flag_rate is None]
        if unmeasured:
            found.append(
                f"{self.attribute}: {len(unmeasured)} group(s) had too few "
                f"reviewed donations to measure ({', '.join(sorted(unmeasured)[:5])})"
            )
        return tuple(found)

    def describe(self) -> str:
        lines = [f"by {self.attribute}:"]
        lines.extend(f"  {group.describe()}" for group in self.groups)
        if self.disparity is not None:
            lines.append(f"  disparity {self.disparity:.2f}x")
        else:
            lines.append(f"  disparity not measurable — {self.unmeasurable_reason}")
        return "\n".join(lines)


def _performance(group: str, members: Sequence[Cohort]) -> GroupPerformance:
    reviewed = [m for m in members if m.reviewed]
    flagged = sum(1 for m in members if m.flagged)
    confirmed = sum(1 for m in reviewed if m.confirmed_risky)

    if len(reviewed) < MIN_GROUP_REVIEWED:
        return GroupPerformance(
            group=group,
            total=len(members),
            reviewed=len(reviewed),
            flagged=flagged,
            confirmed_risky=confirmed,
            precision=None,
            false_flag_rate=None,
            detection_rate=None,
            false_flag_interval=None,
            unmeasurable_reason=(
                f"{len(reviewed)} reviewed donations, below the {MIN_GROUP_REVIEWED} "
                f"needed for a rate that is not sampling noise"
            ),
        )

    clean = [m for m in reviewed if not m.confirmed_risky]
    risky = [m for m in reviewed if m.confirmed_risky]
    flagged_and_reviewed = [m for m in reviewed if m.flagged]

    false_flags = sum(1 for m in clean if m.flagged)
    false_flag_rate = false_flags / len(clean) if clean else None
    interval = wilson_interval(false_flags, len(clean)) if clean else None

    return GroupPerformance(
        group=group,
        total=len(members),
        reviewed=len(reviewed),
        flagged=flagged,
        confirmed_risky=confirmed,
        precision=(
            sum(1 for m in flagged_and_reviewed if m.confirmed_risky)
            / len(flagged_and_reviewed)
            if flagged_and_reviewed
            else None
        ),
        false_flag_rate=false_flag_rate,
        detection_rate=(
            sum(1 for m in risky if m.flagged) / len(risky) if risky else None
        ),
        false_flag_interval=interval,
        unmeasurable_reason=(
            None
            if false_flag_rate is not None
            else "no reviewed donation in this group was confirmed clean, so "
            "there is nothing a false flag could have been measured against"
        ),
    )


def _attribute_key(attribute: str) -> Callable[[Cohort], str | None]:
    if attribute == "size_band":
        return lambda c: c.band
    return lambda c: getattr(c, attribute, None)


def differential_performance(
    cohorts: Sequence[Cohort], *, attribute: str
) -> DifferentialReport:
    """Break performance down by one attribute and measure the spread.

    Donations whose attribute is unknown form their own group rather than being
    dropped. Dropping them would hide the case where the unknowns are
    systematically one district whose records digitise badly, which is a real
    pattern and one this system would otherwise create.
    """
    key = _attribute_key(attribute)
    buckets: dict[str, list[Cohort]] = defaultdict(list)
    for cohort in cohorts:
        value = key(cohort)
        buckets[value if value is not None else "unknown"].append(cohort)

    groups = tuple(
        sorted(
            (_performance(name, members) for name, members in buckets.items()),
            key=lambda g: g.group,
        )
    )

    rated = [g for g in groups if g.false_flag_rate is not None]
    if len(rated) < 2:
        return DifferentialReport(
            attribute=attribute,
            groups=groups,
            disparity=None,
            extremes=None,
            unmeasurable_reason=(
                f"{len(rated)} of {len(groups)} group(s) had enough reviewed "
                f"donations to measure; a disparity needs at least two"
            ),
        )

    highest = max(rated, key=lambda g: g.false_flag_rate or 0.0)
    lowest = min(rated, key=lambda g: g.false_flag_rate or 0.0)
    if not lowest.false_flag_rate:
        # A zero denominator for the ratio. Reported as unmeasurable rather
        # than infinite: "infinitely worse" is not a finding anyone can act on,
        # and the absolute rates are already in the group lines.
        return DifferentialReport(
            attribute=attribute,
            groups=groups,
            disparity=None,
            extremes=(highest.group, lowest.group),
            unmeasurable_reason=(
                f"{lowest.group} had no false flags at all, so the ratio has no "
                f"denominator; compare the absolute rates instead"
            ),
        )

    return DifferentialReport(
        attribute=attribute,
        groups=groups,
        disparity=(highest.false_flag_rate or 0.0) / lowest.false_flag_rate,
        extremes=(highest.group, lowest.group),
    )


def cramers_v(cohorts: Sequence[Cohort]) -> float | None:
    """Association between affiliation and being flagged, on a 0–1 scale.

    Chi-square scales with sample size, so a large enough sample makes any
    trivial association "significant". Cramér's V does not, which is what is
    wanted: the question is how strongly affiliation and flagging move together,
    not whether the link is distinguishable from exactly zero.

    Returns None when the table is degenerate — one affiliation, or nothing
    flagged — because an association needs two things to associate.
    """
    known = [c for c in cohorts if c.affiliation is not None]
    if not known:
        return None

    parties = sorted({c.affiliation for c in known if c.affiliation})
    if len(parties) < 2:
        return None

    flagged_by_party = {
        party: sum(1 for c in known if c.affiliation == party and c.flagged)
        for party in parties
    }
    total_by_party = {
        party: sum(1 for c in known if c.affiliation == party) for party in parties
    }
    total = len(known)
    total_flagged = sum(flagged_by_party.values())
    if total_flagged in (0, total):
        return None

    chi_square = 0.0
    for party in parties:
        for observed, expected in (
            (
                flagged_by_party[party],
                total_by_party[party] * total_flagged / total,
            ),
            (
                total_by_party[party] - flagged_by_party[party],
                total_by_party[party] * (total - total_flagged) / total,
            ),
        ):
            if expected > 0:
                chi_square += (observed - expected) ** 2 / expected

    # For a k x 2 table the normalising term is min(k - 1, 1) = 1.
    return math.sqrt(chi_square / total)


@dataclass(frozen=True)
class AffiliationReport:
    """Whether the output tracks which party received the donation."""

    #: Flag rate per party. Description; parties legitimately differ.
    selection_rates: Mapping[str, float]
    #: Association between affiliation and being flagged, 0–1, or None when the
    #: table cannot support one.
    association: float | None
    #: Differential *error* rates by affiliation. The part that gates.
    errors: DifferentialReport
    #: Share of donations whose recipient affiliation is unknown. A high share
    #: makes every figure above a statement about a subset.
    unknown_affiliation_share: float
    unmeasurable_reason: str | None = None

    @property
    def acceptable(self) -> bool | None:
        """Whether the model may serve on fairness grounds.

        None when the assessment could not be made. An unevaluated fairness
        check is not a passed one, and the promotion gate treats it as blocking.
        """
        return self.errors.within_tolerance

    def concerns(self) -> tuple[str, ...]:
        found = list(self.errors.concerns())
        if self.unknown_affiliation_share > 0.20:
            found.append(
                f"affiliation is unknown for "
                f"{self.unknown_affiliation_share:.0%} of donations; the rates "
                f"above describe the remainder, not the population"
            )
        if self.association is not None and self.association >= 0.20:
            found.append(
                f"flagging and affiliation are associated at V={self.association:.2f}; "
                f"this is not itself a defect — one party may genuinely receive "
                f"more risky donations — but it must be explained before serving"
            )
        return tuple(found)

    def describe(self) -> str:
        lines = ["affiliation assessment:"]
        for party, rate in sorted(self.selection_rates.items()):
            lines.append(f"  {party}: flagged {rate:.3f}")
        lines.append(
            f"  association V={self.association:.3f}"
            if self.association is not None
            else "  association not measurable"
        )
        lines.append(self.errors.describe())
        for concern in self.concerns():
            lines.append(f"  ! {concern}")
        return "\n".join(lines)


def affiliation_assessment(cohorts: Sequence[Cohort]) -> AffiliationReport:
    """The MUST-level neutrality check, run before promotion and quarterly.

    It cannot be waived on the grounds that affiliation is not a model input.
    That is the condition under which it is run, not a reason to skip it.
    """
    if not cohorts:
        return AffiliationReport(
            selection_rates={},
            association=None,
            errors=DifferentialReport(
                attribute="affiliation",
                groups=(),
                disparity=None,
                extremes=None,
                unmeasurable_reason="no donations supplied",
            ),
            unknown_affiliation_share=0.0,
            unmeasurable_reason="no donations supplied",
        )

    known = [c for c in cohorts if c.affiliation is not None]
    rates: dict[str, float] = {}
    for party in sorted({c.affiliation for c in known if c.affiliation}):
        members = [c for c in known if c.affiliation == party]
        rates[party] = sum(1 for m in members if m.flagged) / len(members)

    return AffiliationReport(
        selection_rates=rates,
        association=cramers_v(cohorts),
        errors=differential_performance(cohorts, attribute="affiliation"),
        unknown_affiliation_share=1.0 - len(known) / len(cohorts),
    )


#: The breakdowns taken on every assessment. Fixed rather than configurable:
#: a fairness report whose dimensions are chosen per run is a report whose
#: dimensions get chosen to look good.
REQUIRED_BREAKDOWNS: tuple[str, ...] = ("district", "recipient_type", "size_band")


@dataclass(frozen=True)
class FairnessReport:
    """Everything the promotion gate needs, in one object."""

    affiliation: AffiliationReport
    breakdowns: tuple[DifferentialReport, ...]
    assessed_at: str | None = None
    assessed_by: str | None = None

    @property
    def passed(self) -> bool | None:
        """True only when every dimension was measured and within tolerance.

        A dimension that could not be measured makes the whole assessment
        None — unevaluated — rather than dragging it to False. The distinction
        matters: False means a disparity was found, None means nobody knows,
        and they call for different responses even though both block.
        """
        verdicts = [self.affiliation.acceptable] + [
            report.within_tolerance for report in self.breakdowns
        ]
        if any(verdict is False for verdict in verdicts):
            return False
        if any(verdict is None for verdict in verdicts):
            return None
        return True

    def concerns(self) -> tuple[str, ...]:
        found = list(self.affiliation.concerns())
        for report in self.breakdowns:
            found.extend(report.concerns())
        return tuple(found)

    def describe(self) -> str:
        lines = [self.affiliation.describe()]
        lines.extend(report.describe() for report in self.breakdowns)
        verdict = {True: "within tolerance", False: "DISPARITY FOUND", None: "not assessable"}[
            self.passed
        ]
        lines.append("")
        lines.append(f"fairness: {verdict}")
        return "\n".join(lines)


def assess(
    cohorts: Sequence[Cohort],
    *,
    assessed_at: str | None = None,
    assessed_by: str | None = None,
) -> FairnessReport:
    """Run the full assessment: affiliation plus every required breakdown."""
    return FairnessReport(
        affiliation=affiliation_assessment(cohorts),
        breakdowns=tuple(
            differential_performance(cohorts, attribute=attribute)
            for attribute in REQUIRED_BREAKDOWNS
        ),
        assessed_at=assessed_at,
        assessed_by=assessed_by,
    )
