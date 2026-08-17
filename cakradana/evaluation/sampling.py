"""Random audit sampling, and the recall estimate that depends on it.

Every other metric in this package is computed over donations somebody looked
at, and somebody looked at them because the system surfaced them. That makes
the confirmed-risky set a sample of the system's own output: it can say how
often a flag was right, and it cannot say anything at all about what was
missed. Precision is measurable from the queue. Recall is not.

The fix is not statistical cleverness. It is reviewing a random sample of the
donations the system did *not* flag, every period, and paying for it in analyst
hours. That sample is the only unbiased view of the false negatives, and
without it a detection-rate claim has nothing behind it.

So this module does two things. It draws the sample reproducibly, and it
refuses to return a recall figure when the sample is absent or too small to
support one — an interval so wide it admits both 0.1 and 0.9 is not a
measurement, and publishing its midpoint is worse than publishing nothing.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Sequence

#: Share of the unflagged population reviewed each period. Small enough to be
#: affordable, large enough that the interval means something at the volumes in
#: `NFR-08`. It is an operational cost, not a modelling parameter.
DEFAULT_AUDIT_FRACTION = 0.02

#: Below this many reviewed items the estimate is not reported. At n=20 and one
#: hit, the interval spans most of the unit interval; a midpoint drawn from it
#: would be quoted as a detection rate and would mean nothing.
MIN_AUDIT_SAMPLE = 50

#: Standard normal quantile for a 95% interval.
Z_95 = 1.959963985


@dataclass(frozen=True)
class AuditDraw:
    """A sample of unflagged donations selected for review."""

    period: str
    #: Donations to review. Selected without regard to score — that is the
    #: entire point, and any prioritisation reintroduces the bias the sample
    #: exists to escape.
    donation_ids: tuple[str, ...]
    population_size: int
    fraction: float
    seed: int

    @property
    def size(self) -> int:
        return len(self.donation_ids)

    @property
    def selection_probability(self) -> float:
        if self.population_size == 0:
            return 0.0
        return self.size / self.population_size


class AuditSampler:
    """Draws the unflagged sample for a period.

    Seeded per period, so the same period always yields the same sample. A
    reviewer who reopens the period sees the work they were assigned rather
    than a fresh draw, and an auditor can reconstruct what was reviewed.
    """

    def __init__(
        self,
        *,
        fraction: float = DEFAULT_AUDIT_FRACTION,
        minimum: int = MIN_AUDIT_SAMPLE,
    ) -> None:
        if not 0 < fraction <= 1:
            raise ValueError("audit fraction must lie in (0, 1]")
        self.fraction = fraction
        self.minimum = minimum

    def draw(self, unflagged_ids: Sequence[str], *, period: str) -> AuditDraw:
        """Select donations to review from the ones nothing flagged."""
        population = sorted(set(unflagged_ids))
        # Deterministic in the period rather than in the wall clock, so the
        # draw is reproducible from the record of what it was drawn for.
        # Hashed with SHA-256 rather than the built-in hash, which is salted
        # per process: the same period would otherwise yield a different sample
        # on every restart, and the reproducibility claimed here would be
        # false in exactly the way nobody checks.
        seed = int.from_bytes(hashlib.sha256(period.encode()).digest()[:4], "big")
        rng = random.Random(seed)

        target = min(len(population), max(self.minimum, round(len(population) * self.fraction)))
        selected = rng.sample(population, target) if population else []
        return AuditDraw(
            period=period,
            donation_ids=tuple(sorted(selected)),
            population_size=len(population),
            fraction=self.fraction,
            seed=seed,
        )


@dataclass(frozen=True)
class AuditFinding:
    """The outcome of reviewing one sampled donation."""

    donation_id: str
    confirmed_risky: bool


@dataclass(frozen=True)
class RecallEstimate:
    """What can be said about detection rate, including when nothing can.

    A null `value` with a stated reason is a valid and expected result. It is
    the honest output whenever the sample is missing or too small, and it is
    reported rather than substituted with a figure from the flagged population,
    which would measure the system against its own output.
    """

    value: float | None
    lower: float | None
    upper: float | None
    method: str
    #: Why no figure is given, when none is.
    unmeasurable_reason: str | None = None
    #: Confirmed risky among the donations the system surfaced.
    detected: int = 0
    #: Estimated confirmed risky among the donations it did not.
    missed_estimate: float | None = None
    sample_size: int = 0
    unflagged_population: int = 0

    @property
    def is_measurable(self) -> bool:
        return self.value is not None

    def describe(self) -> str:
        if not self.is_measurable:
            return f"recall not measurable: {self.unmeasurable_reason}"
        return (
            f"recall {self.value:.3f} (95% CI {self.lower:.3f}–{self.upper:.3f}) "
            f"by {self.method}, n={self.sample_size}"
        )


def estimate_recall(
    *,
    detected_risky: int,
    audit_findings: Sequence[AuditFinding],
    unflagged_population: int,
    minimum_sample: int = MIN_AUDIT_SAMPLE,
) -> RecallEstimate:
    """Estimate what share of risky donations the system surfaced.

    The missed count is not observed; it is projected from the audit sample
    onto the unflagged population, and the uncertainty in that projection is
    carried through to the interval rather than dropped. A point estimate with
    no interval invites being quoted as a fact.
    """
    n = len(audit_findings)
    if n == 0:
        return RecallEstimate(
            value=None,
            lower=None,
            upper=None,
            method="none",
            unmeasurable_reason=(
                "no random audit sample was reviewed for this period, so the "
                "donations the system did not flag are unobserved and the "
                "denominator of recall is unknown"
            ),
            detected=detected_risky,
            unflagged_population=unflagged_population,
        )

    if n < minimum_sample:
        return RecallEstimate(
            value=None,
            lower=None,
            upper=None,
            method="audit_sample",
            unmeasurable_reason=(
                f"only {n} unflagged donations were reviewed; at least "
                f"{minimum_sample} are needed before the projection onto "
                f"{unflagged_population} unflagged records carries a usable "
                f"interval"
            ),
            detected=detected_risky,
            sample_size=n,
            unflagged_population=unflagged_population,
        )

    hits = sum(1 for finding in audit_findings if finding.confirmed_risky)
    rate = hits / n
    lower_rate, upper_rate = wilson_interval(hits, n)

    # Projected onto the population the sample was drawn from. The bounds are
    # projected too: a rate uncertain by a factor of three describes a missed
    # count uncertain by a factor of three, and the recall figure inherits it.
    missed = rate * unflagged_population
    missed_low = lower_rate * unflagged_population
    missed_high = upper_rate * unflagged_population

    def recall_for(missed_count: float) -> float:
        total = detected_risky + missed_count
        return detected_risky / total if total > 0 else 0.0

    return RecallEstimate(
        value=recall_for(missed),
        # More missed means lower recall, so the interval inverts.
        lower=recall_for(missed_high),
        upper=recall_for(missed_low),
        method="audit_sample",
        detected=detected_risky,
        missed_estimate=missed,
        sample_size=n,
        unflagged_population=unflagged_population,
    )


def wilson_interval(successes: int, trials: int, *, z: float = Z_95) -> tuple[float, float]:
    """A binomial interval that behaves at small counts and near zero.

    The textbook normal approximation produces negative lower bounds when the
    observed rate is near zero, which is exactly the regime an audit sample
    operates in: most sampled donations are fine. An interval that runs below
    zero is a sign the method does not apply, not a small rounding problem.
    """
    if trials == 0:
        return (0.0, 1.0)
    p = successes / trials
    denominator = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denominator
    margin = (
        z
        * math.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2))
        / denominator
    )
    return (max(0.0, centre - margin), min(1.0, centre + margin))


@dataclass(frozen=True)
class SamplingBias:
    """How far the reviewed population departs from the whole one.

    Reported whenever recall is, because a precision figure computed over a
    queue is a statement about the queue. Saying so is the third of the three
    mitigations and the one that always applies.
    """

    reviewed_flagged: int
    reviewed_unflagged: int
    flagged_population: int
    unflagged_population: int

    @property
    def flagged_review_rate(self) -> float:
        if self.flagged_population == 0:
            return 0.0
        return self.reviewed_flagged / self.flagged_population

    @property
    def unflagged_review_rate(self) -> float:
        if self.unflagged_population == 0:
            return 0.0
        return self.reviewed_unflagged / self.unflagged_population

    @property
    def ratio(self) -> float | None:
        """How many times more likely a flagged donation was to be reviewed.

        None when no unflagged donation was reviewed at all — the ratio is not
        infinite, it is undefined, and the distinction matters because the
        first reads as a very large number and the second as a missing control.
        """
        if self.unflagged_review_rate == 0:
            return None
        return self.flagged_review_rate / self.unflagged_review_rate

    def describe(self) -> str:
        if self.ratio is None:
            return (
                "no unflagged donations were reviewed; every human label "
                "describes a donation the system surfaced, and no statement "
                "about what it missed is supported"
            )
        return (
            f"a flagged donation was {self.ratio:.1f}× more likely to be "
            f"reviewed than an unflagged one; precision figures describe the "
            f"reviewed population, not the whole one"
        )


def propensity_weights(
    reviewed: Sequence[tuple[str, float]],
) -> dict[str, float]:
    """Inverse-probability weights over the flagged population.

    The second mitigation, used when a random audit sample is not available.
    Each reviewed donation stands in for however many similar donations were
    not reviewed. It corrects for unequal selection among donations that could
    have been selected; it cannot say anything about donations with zero
    probability of selection, which is precisely the blind spot the audit
    sample exists to cover.
    """
    weights: dict[str, float] = {}
    for donation_id, probability in reviewed:
        if probability <= 0:
            raise ValueError(
                f"{donation_id} has a zero selection probability; it could not "
                f"have been reviewed, and weighting cannot recover a stratum "
                f"that was never sampled"
            )
        weights[donation_id] = 1.0 / probability
    return weights
