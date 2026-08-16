"""Dataset splitting.

Splits are grouped by donor, and donor overlap is asserted to be zero rather
than measured and tolerated. The same donor on both sides lets the model
recognise them instead of generalising, and donor behaviour is most of what
these features describe. A split that quietly leaks a few hundred donors
produces a number nobody can distinguish from an honest one.

Donors are assigned to cohorts by when each first appeared, so the cohorts are
also ordered in time. What this measures is generalisation to donors the model
has never seen, which is the question serving actually asks of it.

It does not measure forecasting a later period from an earlier one. Cutting on
a date and discarding donors who straddle the boundary would, and it is not
usable here: donors give across a whole year, so nearly every donor active late
is also active early, and that filter discards the overwhelming majority of the
later data. The alternative to this trade-off is not a stricter split, it is a
test set of a few dozen donations from which no metric means anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from cakradana.schema import Donation


class LeakageError(AssertionError):
    """Raised when a split would let information cross the boundary."""


@dataclass(frozen=True)
class Split:
    """One partition of the data."""

    name: str
    donations: tuple[Donation, ...]

    def __len__(self) -> int:
        return len(self.donations)

    @property
    def donors(self) -> set[str]:
        return {
            d.sender_ref.entity_id
            for d in self.donations
            if d.sender_ref.entity_id is not None
        }

    @property
    def span(self) -> tuple[datetime, datetime] | None:
        if not self.donations:
            return None
        stamps = [d.occurred_at for d in self.donations]
        return min(stamps), max(stamps)


@dataclass(frozen=True)
class SplitSet:
    train: Split
    calibration: Split
    test: Split

    def summary(self) -> dict[str, object]:
        return {
            split.name: {
                "donations": len(split),
                "donors": len(split.donors),
                "span": [s.isoformat() for s in split.span] if split.span else None,
            }
            for split in (self.train, self.calibration, self.test)
        }


def donor_cohort_split(
    donations: Sequence[Donation],
    *,
    train_fraction: float = 0.6,
    calibration_fraction: float = 0.15,
) -> SplitSet:
    """Split donors into cohorts by when each first appeared.

    Every donation follows its donor, so the splits are donor-disjoint by
    construction rather than by filtering, and cohorts are ordered in time.
    This measures the question deployment actually asks: given a donor the
    model has never seen, does it generalise?

    Cutting purely on a date and then discarding donors who straddle the
    boundary sounds stricter and is not usable here. Donors give across a whole
    year, so almost every donor active late is also active early, and the
    filter throws away the great majority of the later data — leaving a test
    set of a few dozen donations, from which no metric means anything.

    The trade-off is explicit: a test donor's donations may fall early in the
    period, so this does not measure forecasting a later time from an earlier
    one. It measures generalisation to unseen donors, which is what the model
    is asked to do at serving time.
    """
    if not donations:
        raise ValueError("cannot split an empty dataset")
    if train_fraction + calibration_fraction >= 1.0:
        raise ValueError("no data would remain for the test split")

    first_seen: dict[str, datetime] = {}
    for donation in donations:
        donor = donation.sender_ref.entity_id
        if donor is None:
            continue
        stamp = donation.occurred_at
        if donor not in first_seen or stamp < first_seen[donor]:
            first_seen[donor] = stamp

    if not first_seen:
        raise ValueError("no donation carries a resolved donor to group by")

    ordered_donors = sorted(first_seen, key=lambda d: (first_seen[d], d))
    n = len(ordered_donors)
    train_end = max(int(n * train_fraction), 1)
    calibration_end = max(int(n * (train_fraction + calibration_fraction)), train_end + 1)

    assignment: dict[str, str] = {}
    for index, donor in enumerate(ordered_donors):
        if index < train_end:
            assignment[donor] = "train"
        elif index < calibration_end:
            assignment[donor] = "calibration"
        else:
            assignment[donor] = "test"

    buckets: dict[str, list[Donation]] = {"train": [], "calibration": [], "test": []}
    for donation in donations:
        donor = donation.sender_ref.entity_id
        if donor is None:
            # An unresolved donor cannot be held out, because there is no
            # identity to hold out. These train the model and are kept out of
            # evaluation rather than being counted as a clean generalisation.
            buckets["train"].append(donation)
        else:
            buckets[assignment[donor]].append(donation)

    splits = SplitSet(
        train=Split("train", tuple(_by_time(buckets["train"]))),
        calibration=Split("calibration", tuple(_by_time(buckets["calibration"]))),
        test=Split("test", tuple(_by_time(buckets["test"]))),
    )
    assert_no_leakage(splits)
    return splits


def _by_time(donations: list[Donation]) -> list[Donation]:
    return sorted(donations, key=lambda d: (d.occurred_at, d.donation_id))


def assert_no_leakage(splits: SplitSet) -> None:
    """Fail loudly on any donor appearing in more than one split.

    Asserted rather than measured. A leakage rate that is merely reported gets
    read, noted, and lived with, and the resulting metric is indistinguishable
    from an honest one.
    """
    pairs = (
        ("train", "calibration", splits.train, splits.calibration),
        ("train", "test", splits.train, splits.test),
        ("calibration", "test", splits.calibration, splits.test),
    )
    for left_name, right_name, left, right in pairs:
        overlap = left.donors & right.donors
        if overlap:
            raise LeakageError(
                f"{len(overlap)} donors appear in both {left_name} and "
                f"{right_name}; the model would be recognising donors rather "
                f"than generalising"
            )

    # Spans overlap by design: cohorts are separated by donor, not by date, so
    # a test donor's donations may fall anywhere in the period. What must not
    # happen — a donor appearing on both sides — is checked above.
