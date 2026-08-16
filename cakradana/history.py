"""Point-in-time donation history.

Every aggregate in this system is computed through a view bound to a single
timestamp. A donation contributes to a view only when it both occurred and was
recorded at or before that timestamp.

Both conditions are necessary and the second is the one that is easy to lose. A
donation that occurred in January but was scraped in June was not knowable in
February, and including it in a February aggregate leaks future information
into a past decision. Computing aggregates over a whole dataset before
splitting it is the same mistake at training time, and it is what made earlier
measurements of this system meaningless.

The filter lives here, in one place, rather than in each caller, so that
forgetting it is not possible.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Iterable, Iterator, Protocol, Sequence

from cakradana.schema import Donation


class DonationStore(Protocol):
    """Read interface the rule engine and feature service depend on.

    Kept narrow so that an in-memory replay store and a database-backed store
    are interchangeable, and so that point-in-time correctness is a property of
    the view rather than of any particular backend.
    """

    def knowable_at(self, as_of: datetime) -> PointInTimeView: ...


class DonationIndex:
    """Donations grouped by counterparty, each group ordered by occurrence.

    Built once and shared by every view over the same store. Rebuilding these
    groups per view turns a backfill into quadratic work — a replay of a
    hundred thousand donations would rebuild a hundred thousand indexes of a
    hundred thousand entries each — and a training run that cannot finish is a
    training run nobody does.
    """

    __slots__ = ("donations", "by_sender", "by_receiver", "by_pair")

    def __init__(self, donations: Sequence[Donation]) -> None:
        self.donations = tuple(sorted(donations, key=lambda d: d.occurred_at))
        self.by_sender: dict[str, list[Donation]] = defaultdict(list)
        self.by_receiver: dict[str, list[Donation]] = defaultdict(list)
        self.by_pair: dict[tuple[str, str], list[Donation]] = defaultdict(list)

        for donation in self.donations:
            sender = donation.sender_ref.entity_id
            receiver = donation.receiver_ref.entity_id
            if sender is not None:
                self.by_sender[sender].append(donation)
            if receiver is not None:
                self.by_receiver[receiver].append(donation)
            if sender is not None and receiver is not None:
                self.by_pair[(sender, receiver)].append(donation)


class PointInTimeView:
    """Donations the system could have known about at ``as_of``.

    A view is a bounded window onto a shared index rather than a copy. Lookups
    binary-search the relevant group for the cutoff and then drop anything the
    system had not yet been told about, so the cost of a query scales with one
    counterparty's history rather than with the whole dataset.
    """

    __slots__ = ("_as_of", "_index")

    def __init__(
        self, donations: Sequence[Donation] | DonationIndex, as_of: datetime
    ) -> None:
        self._as_of = as_of
        self._index = (
            donations
            if isinstance(donations, DonationIndex)
            else DonationIndex(donations)
        )

    @property
    def as_of(self) -> datetime:
        return self._as_of

    def _visible(self, donations: Sequence[Donation]) -> tuple[Donation, ...]:
        return self._window(donations, None, None, None)

    def __len__(self) -> int:
        return len(self._visible(self._index.donations))

    def __iter__(self) -> Iterator[Donation]:
        return iter(self._visible(self._index.donations))

    # -- lookups ---------------------------------------------------------
    #
    # Each takes an optional window and an ``excluding`` donation id. Rules
    # that test a cumulative total include the donation being scored, because
    # the finding attaches to the donation that crossed the threshold. Features
    # describing a donor's prior behaviour exclude it, because a feature that
    # can see its own donation has seen the future.

    def by_sender(
        self,
        sender_id: str,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        excluding: str | None = None,
    ) -> tuple[Donation, ...]:
        return self._window(
            self._index.by_sender.get(sender_id, ()), since, until, excluding
        )

    def by_receiver(
        self,
        receiver_id: str,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        excluding: str | None = None,
    ) -> tuple[Donation, ...]:
        return self._window(
            self._index.by_receiver.get(receiver_id, ()), since, until, excluding
        )

    def by_pair(
        self,
        sender_id: str,
        receiver_id: str,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        excluding: str | None = None,
    ) -> tuple[Donation, ...]:
        return self._window(
            self._index.by_pair.get((sender_id, receiver_id), ()),
            since,
            until,
            excluding,
        )

    def _window(
        self,
        donations: Sequence[Donation],
        since: datetime | None,
        until: datetime | None,
        excluding: str | None,
    ) -> tuple[Donation, ...]:
        """Slice a group to a window and to what was knowable at the cutoff.

        Groups are kept in occurrence order, so both ends of the window are
        found by binary search rather than by scanning. What remains is
        filtered on recording time in the same pass: a donation that happened
        in January but only reached the system in June was not knowable in
        February, and admitting it is the leak that makes a measurement
        impossible to reproduce at serving time.
        """
        if not donations:
            return ()

        cutoff = self._as_of if until is None else min(until, self._as_of)
        upper = bisect_right(donations, cutoff, key=lambda d: d.occurred_at)
        lower = (
            bisect_left(donations, since, key=lambda d: d.occurred_at)
            if since is not None
            else 0
        )
        if lower >= upper:
            return ()

        as_of = self._as_of
        result: Iterable[Donation] = donations[lower:upper]
        result = (d for d in result if d.recorded_at <= as_of)
        if excluding is not None:
            result = (d for d in result if d.donation_id != excluding)
        return tuple(result)

    # -- derived quantities ----------------------------------------------

    def sender_first_seen(self, sender_id: str) -> datetime | None:
        history = self._window(
            self._index.by_sender.get(sender_id, ()), None, None, None
        )
        return history[0].occurred_at if history else None

    def distinct_senders_to(
        self, receiver_id: str, *, since: datetime | None = None
    ) -> set[str]:
        return {
            d.sender_ref.entity_id
            for d in self.by_receiver(receiver_id, since=since)
            if d.sender_ref.entity_id is not None
        }

    def distinct_receivers_from(
        self, sender_id: str, *, since: datetime | None = None
    ) -> set[str]:
        return {
            d.receiver_ref.entity_id
            for d in self.by_sender(sender_id, since=since)
            if d.receiver_ref.entity_id is not None
        }

    def has_prior_history(self, sender_id: str, *, before: datetime) -> bool:
        """Whether a donor donated before ``before``.

        Used by the fan-in heuristic: a burst of donations from donors with no
        history is materially more suspicious than the same burst from
        established donors, and treating those cases alike is what makes a
        naive fan-in detector unusable on real grassroots fundraising.
        """
        first = self.sender_first_seen(sender_id)
        # Groups are kept in occurrence order, so the earliest visible donation
        # decides this without scanning.
        return first is not None and first < before


class InMemoryDonationStore:
    """Replay store.

    Backs training backfill, tests, and the rule fixtures. Serving uses a
    database-backed store exposing the same interface.
    """

    def __init__(self, donations: Iterable[Donation] = ()) -> None:
        self._donations: list[Donation] = list(donations)
        self._cached: DonationIndex | None = None
        self._cached_size = -1

    def add(self, donation: Donation) -> None:
        self._donations.append(donation)

    def extend(self, donations: Iterable[Donation]) -> None:
        self._donations.extend(donations)

    def __len__(self) -> int:
        return len(self._donations)

    def knowable_at(self, as_of: datetime) -> PointInTimeView:
        return PointInTimeView(self._index(), as_of)

    def _index(self) -> DonationIndex:
        """Build the shared index once and reuse it until the store changes."""
        if self._cached is None or self._cached_size != len(self._donations):
            self._cached = DonationIndex(self._donations)
            self._cached_size = len(self._donations)
        return self._cached

    def replay(self) -> Iterator[tuple[Donation, PointInTimeView]]:
        """Yield each donation with the view that was current when it arrived.

        Ordered by ``recorded_at``, which is the order the system learned
        things, so that a backfill reproduces the sequence of states serving
        would have passed through. Iterating in ``occurred_at`` order instead
        would hand each donation a view containing records that had not yet
        been received.
        """
        index = self._index()
        for donation in sorted(
            self._donations, key=lambda d: (d.recorded_at, d.donation_id)
        ):
            yield donation, PointInTimeView(index, donation.occurred_at)


def days_before(when: datetime, days: int) -> datetime:
    return when - timedelta(days=days)
