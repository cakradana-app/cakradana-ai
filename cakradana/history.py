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


class PointInTimeView:
    """Donations the system could have known about at ``as_of``.

    Instances are cheap to create and safe to hold: they never observe
    donations added to the underlying store afterwards.
    """

    def __init__(self, donations: Sequence[Donation], as_of: datetime) -> None:
        self._as_of = as_of
        self._donations = tuple(
            d
            for d in donations
            if d.occurred_at <= as_of and d.recorded_at <= as_of
        )
        self._by_sender: dict[str, list[Donation]] = defaultdict(list)
        self._by_receiver: dict[str, list[Donation]] = defaultdict(list)
        self._by_pair: dict[tuple[str, str], list[Donation]] = defaultdict(list)

        for donation in sorted(self._donations, key=lambda d: d.occurred_at):
            sender = donation.sender_ref.entity_id
            receiver = donation.receiver_ref.entity_id
            if sender is not None:
                self._by_sender[sender].append(donation)
            if receiver is not None:
                self._by_receiver[receiver].append(donation)
            if sender is not None and receiver is not None:
                self._by_pair[(sender, receiver)].append(donation)

    @property
    def as_of(self) -> datetime:
        return self._as_of

    def __len__(self) -> int:
        return len(self._donations)

    def __iter__(self) -> Iterator[Donation]:
        return iter(self._donations)

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
        return self._window(self._by_sender.get(sender_id, ()), since, until, excluding)

    def by_receiver(
        self,
        receiver_id: str,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
        excluding: str | None = None,
    ) -> tuple[Donation, ...]:
        return self._window(
            self._by_receiver.get(receiver_id, ()), since, until, excluding
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
            self._by_pair.get((sender_id, receiver_id), ()), since, until, excluding
        )

    @staticmethod
    def _window(
        donations: Sequence[Donation],
        since: datetime | None,
        until: datetime | None,
        excluding: str | None,
    ) -> tuple[Donation, ...]:
        result: Iterable[Donation] = donations
        if since is not None:
            result = (d for d in result if d.occurred_at >= since)
        if until is not None:
            result = (d for d in result if d.occurred_at <= until)
        if excluding is not None:
            result = (d for d in result if d.donation_id != excluding)
        return tuple(result)

    # -- derived quantities ----------------------------------------------

    def sender_first_seen(self, sender_id: str) -> datetime | None:
        history = self._by_sender.get(sender_id)
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
        history = self._by_sender.get(sender_id)
        # Per-sender lists are built in occurred_at order, so the earliest
        # donation decides this without scanning.
        return bool(history) and history[0].occurred_at < before


class InMemoryDonationStore:
    """Replay store.

    Backs training backfill, tests, and the rule fixtures. Serving uses a
    database-backed store exposing the same interface.
    """

    def __init__(self, donations: Iterable[Donation] = ()) -> None:
        self._donations: list[Donation] = list(donations)

    def add(self, donation: Donation) -> None:
        self._donations.append(donation)

    def extend(self, donations: Iterable[Donation]) -> None:
        self._donations.extend(donations)

    def __len__(self) -> int:
        return len(self._donations)

    def knowable_at(self, as_of: datetime) -> PointInTimeView:
        return PointInTimeView(self._donations, as_of)

    def replay(self) -> Iterator[tuple[Donation, PointInTimeView]]:
        """Yield each donation with the view that was current when it arrived.

        Ordered by ``recorded_at``, which is the order the system learned
        things, so that a backfill reproduces the sequence of states serving
        would have passed through. Iterating in ``occurred_at`` order instead
        would hand each donation a view containing records that had not yet
        been received.
        """
        for donation in sorted(self._donations, key=lambda d: (d.recorded_at, d.donation_id)):
            yield donation, PointInTimeView(self._donations, donation.occurred_at)


def days_before(when: datetime, days: int) -> datetime:
    return when - timedelta(days=days)
