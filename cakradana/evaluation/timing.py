"""How long scoring takes, and how that changes as history grows.

The latency and scale targets were stated and never measured, which leaves them
as design intent. A target nobody has taken a reading against is not a
commitment; it is a hope with a number on it.

Two things are measured here, and they answer different questions.

**Latency** is a wall-clock figure and belongs to the machine it was taken on.
It is reported with the hardware and the population size, never as a bare
number, because a p95 quoted without its conditions gets repeated in a document
where those conditions are not.

**Scaling** is a property of the code and survives the move to other hardware.
The features are meant to be sub-linear in the size of the history a donation is
judged against; if they are not, the system gets slower as it succeeds, and the
first place anyone notices is production. Measured as a ratio between population
sizes, so the hardware cancels out — which is what makes it the figure worth
asserting in a test.

Percentiles rather than a mean throughout. One donation with an unusually large
donor history takes far longer than the median, and a mean quietly absorbs it:
the tail is what fills a review queue late, so the tail is what is reported.
"""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass
from typing import Callable, Sequence

#: Below this many samples a p95 is one observation, and one observation is not
#: a percentile.
MIN_SAMPLES = 20


def percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile.

    Chosen over interpolation deliberately: every value reported is one that was
    actually observed, so a p95 of 40ms means some request took 40ms rather than
    that a formula produced 40 from two neighbours.
    """
    if not values:
        raise ValueError("no samples")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(fraction * len(ordered)) - 1))
    return ordered[index]


@dataclass(frozen=True)
class LatencyReport:
    """Timings for one operation, with the conditions they were taken under."""

    operation: str
    samples: int
    #: Size of the history each sampled call was judged against.
    population: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    machine: str
    unmeasurable_reason: str | None = None

    @property
    def measured(self) -> bool:
        return self.unmeasurable_reason is None

    def within(self, budget_ms: float) -> bool | None:
        """Whether p95 met a stated budget, or None if nothing was measured.

        None rather than True: a budget nothing was measured against has not
        been met, it has been left unchecked, and the two must not read alike.
        """
        if not self.measured:
            return None
        return self.p95_ms <= budget_ms

    def describe(self) -> str:
        if not self.measured:
            return f"{self.operation}: not measured — {self.unmeasurable_reason}"
        return (
            f"{self.operation}: p50 {self.p50_ms:.1f}ms  p95 {self.p95_ms:.1f}ms  "
            f"p99 {self.p99_ms:.1f}ms  max {self.max_ms:.1f}ms "
            f"({self.samples} samples over a {self.population}-donation history "
            f"on {self.machine})"
        )


def measure(
    operation: str,
    call: Callable[[int], object],
    *,
    samples: int,
    population: int,
    warmup: int = 5,
) -> LatencyReport:
    """Time ``call`` and report its distribution.

    A warm-up runs first and is discarded. The first calls into this system pay
    for module imports, rule-set parsing, and the first pass through code paths
    the interpreter has not seen, none of which a steady-state figure should
    carry.
    """
    if samples < MIN_SAMPLES:
        return LatencyReport(
            operation=operation,
            samples=samples,
            population=population,
            p50_ms=0.0,
            p95_ms=0.0,
            p99_ms=0.0,
            max_ms=0.0,
            machine=_machine(),
            unmeasurable_reason=(
                f"{samples} samples, below the {MIN_SAMPLES} at which a p95 is "
                f"more than a single observation"
            ),
        )

    for index in range(warmup):
        call(index)

    timings: list[float] = []
    for index in range(samples):
        started = time.perf_counter()
        call(index)
        timings.append((time.perf_counter() - started) * 1000.0)

    return LatencyReport(
        operation=operation,
        samples=samples,
        population=population,
        p50_ms=percentile(timings, 0.50),
        p95_ms=percentile(timings, 0.95),
        p99_ms=percentile(timings, 0.99),
        max_ms=max(timings),
        machine=_machine(),
    )


@dataclass(frozen=True)
class ScalingReport:
    """How cost grows with the size of the history being judged against."""

    operation: str
    small: LatencyReport
    large: LatencyReport

    @property
    def population_ratio(self) -> float:
        return self.large.population / max(self.small.population, 1)

    @property
    def cost_ratio(self) -> float | None:
        """How much slower the larger population is, at p95.

        None when either side could not be measured, and None when the smaller
        figure is zero — a ratio against a timing too small to register is not
        a measurement of anything.
        """
        if not (self.small.measured and self.large.measured):
            return None
        if self.small.p95_ms <= 0:
            return None
        return self.large.p95_ms / self.small.p95_ms

    @property
    def is_sublinear(self) -> bool | None:
        """Whether cost grew more slowly than the population did.

        The property that decides whether this system gets slower as it
        succeeds. None when it could not be measured, which blocks the same way
        a failure does and for a different reason.
        """
        ratio = self.cost_ratio
        if ratio is None:
            return None
        return ratio < self.population_ratio

    def describe(self) -> str:
        ratio = self.cost_ratio
        if ratio is None:
            return f"{self.operation}: scaling not measurable"
        verdict = "sub-linear" if self.is_sublinear else "LINEAR OR WORSE"
        return (
            f"{self.operation}: {self.population_ratio:.1f}x the history costs "
            f"{ratio:.2f}x the time — {verdict}\n"
            f"  {self.small.describe()}\n"
            f"  {self.large.describe()}"
        )


def _machine() -> str:
    """Enough to tell two readings apart, and no more.

    Reported with every figure because a latency number without the machine it
    was taken on is not reproducible, and gets quoted anyway.
    """
    return f"{platform.machine()} {platform.python_implementation()} {platform.python_version()}"
