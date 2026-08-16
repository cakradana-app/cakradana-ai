"""Generation-time acceptance checks.

Each typology is recovered by a detector that uses only the signal defining it.
If donation splitting cannot be found by counting converging first-time donors,
then the convergence was never generated and the label is decoration.

This is the check that would have caught the previous generator. Four of its
five typologies were unrecoverable by any means — the illegal-source signal was
deleted before training, self-funded rows were built identically to ordinary
ones, and the splitting and proxy patterns were random amounts with no
structure at all — and nothing in the workflow reported it. A model trained on
that data was fitting noise for most of its positive class, and its scores
looked reasonable throughout.

Detectors here are deliberately crude. They exist to confirm the data contains
what it claims, not to detect anything in production.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta

from cakradana.data.generator import (
    ALL_TYPOLOGIES,
    INDIVIDUAL_PARTY_LIMIT,
    T_CUMULATIVE,
    T_ILLEGAL_SOURCE,
    T_PROXY,
    T_SMURFING,
    T_STRUCTURING,
    SyntheticDataset,
)

#: Minimum share of each typology its own defining signal must recover.
MIN_RECALL = 0.8


@dataclass(frozen=True)
class TypologyCheck:
    typology: str
    generated: int
    recovered: int

    @property
    def recall(self) -> float:
        return self.recovered / self.generated if self.generated else 0.0

    @property
    def passed(self) -> bool:
        return self.generated > 0 and self.recall >= MIN_RECALL

    def describe(self) -> str:
        return (
            f"{self.typology}: {self.recovered}/{self.generated} recovered "
            f"(recall {self.recall:.2f}) {'ok' if self.passed else 'FAILED'}"
        )


class AcceptanceError(AssertionError):
    """Raised when generated data does not contain a typology it claims."""


def check(dataset: SyntheticDataset) -> list[TypologyCheck]:
    detectors = {
        T_SMURFING: _detect_smurfing,
        T_PROXY: _detect_proxy,
        T_STRUCTURING: _detect_structuring,
        T_CUMULATIVE: _detect_cumulative,
        T_ILLEGAL_SOURCE: _detect_illegal_source,
    }
    results = []
    for typology in ALL_TYPOLOGIES:
        expected = {
            did for did, t in dataset.truth.items() if t == typology
        }
        if not expected:
            continue
        found = detectors[typology](dataset)
        results.append(
            TypologyCheck(
                typology=typology,
                generated=len(expected),
                recovered=len(expected & found),
            )
        )
    return results


def assert_acceptable(dataset: SyntheticDataset) -> list[TypologyCheck]:
    results = check(dataset)
    failures = [r for r in results if not r.passed]
    if failures:
        raise AcceptanceError(
            "generated data does not contain the structure it labels:\n  "
            + "\n  ".join(r.describe() for r in failures)
        )
    return results


# ---------------------------------------------------------------------------
# Detectors — one defining signal each
# ---------------------------------------------------------------------------


def _detect_smurfing(dataset: SyntheticDataset) -> set[str]:
    """Donors converging on one recipient in a short window, most of them new."""
    by_receiver: dict[str, list] = defaultdict(list)
    first_seen: dict[str, object] = {}
    for d in sorted(dataset.donations, key=lambda d: d.occurred_at):
        by_receiver[d.receiver_ref.entity_id].append(d)
        first_seen.setdefault(d.sender_ref.entity_id, d.occurred_at)

    ceiling = INDIVIDUAL_PARTY_LIMIT * 0.25
    found: set[str] = set()
    for donations in by_receiver.values():
        for anchor in donations:
            # A seven-day window, matching the scale at which coordinated
            # splitting actually executes. Measuring convergence over a
            # fortnight dilutes a three-day cohort with two weeks of unrelated
            # traffic and hides exactly the pattern being looked for.
            window_start = anchor.occurred_at - timedelta(days=7)
            candidates = [
                d
                for d in donations
                if window_start <= d.occurred_at <= anchor.occurred_at
                and d.amount_idr <= ceiling
                and first_seen[d.sender_ref.entity_id] >= window_start
            ]
            cohort = _largest_homogeneous_group(candidates)
            if len({d.sender_ref.entity_id for d in cohort}) >= 15:
                found.update(d.donation_id for d in cohort)
    return found


def _largest_homogeneous_group(donations: list, tolerance: float = 1.15) -> list:
    """The biggest subset whose amounts sit within a narrow multiple.

    Looking for a homogeneous *subgroup* rather than asking the whole window to
    be homogeneous. A recipient can be running a genuine fundraising drive,
    with the varied amounts real supporters choose, while also receiving a
    split contribution in the same week. Judging the window as a whole averages
    the two together and loses both — the coordination stops looking uniform,
    and the real drive gets tarred by it.
    """
    if not donations:
        return []
    ordered = sorted(donations, key=lambda d: d.amount_idr)
    best: list = []
    start = 0
    for end in range(len(ordered)):
        while ordered[end].amount_idr > ordered[start].amount_idr * tolerance:
            start += 1
        group = ordered[start : end + 1]
        if len(group) > len(best):
            best = group
    return best


def _detect_proxy(dataset: SyntheticDataset) -> set[str]:
    """An entity receiving and forwarding a comparable amount within days."""
    inflows: dict[str, list] = defaultdict(list)
    for d in dataset.donations:
        inflows[d.receiver_ref.entity_id].append(d)

    found: set[str] = set()
    for d in dataset.donations:
        for inflow in inflows.get(d.sender_ref.entity_id, ()):
            lag = (d.occurred_at - inflow.occurred_at).days
            if not 0 <= lag <= 7:
                continue
            ratio = d.amount_idr / inflow.amount_idr
            if 0.85 <= ratio <= 1.15:
                found.add(d.donation_id)
                found.add(inflow.donation_id)
    return found


def _detect_structuring(dataset: SyntheticDataset) -> set[str]:
    """Amounts sitting in the band immediately below the limit."""
    return {
        d.donation_id
        for d in dataset.donations
        if 0.90 <= d.amount_idr / INDIVIDUAL_PARTY_LIMIT <= 0.999
    }


def _detect_cumulative(dataset: SyntheticDataset) -> set[str]:
    """Every donation by a donor whose period total to one recipient is over."""
    totals: dict[tuple[str, str], int] = defaultdict(int)
    grouped: dict[tuple[str, str], list] = defaultdict(list)
    for d in dataset.donations:
        key = (d.sender_ref.entity_id, d.receiver_ref.entity_id)
        totals[key] += d.amount_idr
        grouped[key].append(d)

    found: set[str] = set()
    for key, total in totals.items():
        if total > INDIVIDUAL_PARTY_LIMIT:
            found.update(d.donation_id for d in grouped[key])
    return found


def _detect_illegal_source(dataset: SyntheticDataset) -> set[str]:
    """Donor appears on the prohibited-source register.

    A lookup against register membership carried on the entity — not a match
    against the donor's name, which is what the earlier data relied on and then
    discarded.
    """
    prohibited = {
        e.entity_id
        for e in dataset.entities.values()
        if "prohibited_source" in e.registers
    }
    return {
        d.donation_id
        for d in dataset.donations
        if d.sender_ref.entity_id in prohibited
    }
