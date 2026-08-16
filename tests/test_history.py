"""Point-in-time views.

The single invariant: a view bound to a timestamp never reveals anything the
system did not know by then. Every aggregate in the project is computed through
one of these, so a leak here would be a leak everywhere.
"""

from __future__ import annotations

from cakradana.history import InMemoryDonationStore, PointInTimeView
from tests.conftest import at, make_donation


def test_donations_recorded_later_are_invisible():
    """A donation scraped in June was not knowable in February, even though it
    occurred in January."""
    late_arrival = make_donation(
        donation_id="late", occurred=at(2026, 1, 15), recorded=at(2026, 6, 1)
    )
    store = InMemoryDonationStore([late_arrival])
    assert len(store.knowable_at(at(2026, 2, 1))) == 0
    assert len(store.knowable_at(at(2026, 7, 1))) == 1


def test_donations_occurring_later_are_invisible():
    future = make_donation(donation_id="future", occurred=at(2026, 9, 1))
    store = InMemoryDonationStore([future])
    assert len(store.knowable_at(at(2026, 2, 1))) == 0


def test_a_view_does_not_observe_later_additions():
    store = InMemoryDonationStore([make_donation(donation_id="a")])
    view = store.knowable_at(at(2026, 12, 31))
    store.add(make_donation(donation_id="b", occurred=at(2026, 7, 1)))
    assert len(view) == 1


def test_pair_lookup_excludes_the_donation_being_scored():
    """A feature describing prior behaviour must not see its own donation."""
    first = make_donation(donation_id="a", occurred=at(2026, 1, 1))
    second = make_donation(donation_id="b", occurred=at(2026, 2, 1))
    view = InMemoryDonationStore([first, second]).knowable_at(at(2026, 3, 1))
    prior = view.by_pair("e-sender", "e-receiver", excluding="b")
    assert [d.donation_id for d in prior] == ["a"]


def test_windowing_is_inclusive_of_its_bounds():
    donations = [
        make_donation(donation_id=f"d{i}", occurred=at(2026, 1, i + 1))
        for i in range(5)
    ]
    view = InMemoryDonationStore(donations).knowable_at(at(2026, 6, 1))
    windowed = view.by_sender("e-sender", since=at(2026, 1, 2), until=at(2026, 1, 4))
    assert [d.donation_id for d in windowed] == ["d1", "d2", "d3"]


def test_distinct_counterparties():
    donations = [
        make_donation(donation_id=f"d{i}", sender=f"s{i % 3}", receiver="r1")
        for i in range(6)
    ]
    view = InMemoryDonationStore(donations).knowable_at(at(2026, 6, 1))
    assert view.distinct_senders_to("r1") == {"s0", "s1", "s2"}


def test_prior_history_reflects_the_cutoff():
    donation = make_donation(donation_id="a", occurred=at(2026, 1, 1))
    view = InMemoryDonationStore([donation]).knowable_at(at(2026, 6, 1))
    assert view.has_prior_history("e-sender", before=at(2026, 3, 1))
    assert not view.has_prior_history("e-sender", before=at(2025, 12, 1))
    assert not view.has_prior_history("nobody", before=at(2026, 3, 1))


def test_replay_orders_by_when_the_system_learned_of_each_donation():
    """Replaying in occurred_at order would hand each donation a view holding
    records that had not yet been received, which is the training-time form of
    the same leak."""
    late_arrival = make_donation(
        donation_id="occurred-first", occurred=at(2026, 1, 1), recorded=at(2026, 6, 1)
    )
    prompt = make_donation(
        donation_id="recorded-first", occurred=at(2026, 2, 1), recorded=at(2026, 2, 1)
    )
    store = InMemoryDonationStore([late_arrival, prompt])
    order = [d.donation_id for d, _ in store.replay()]
    assert order == ["recorded-first", "occurred-first"]


def test_replay_view_excludes_the_donation_itself_by_recorded_at():
    late_arrival = make_donation(
        donation_id="late", occurred=at(2026, 1, 1), recorded=at(2026, 6, 1)
    )
    earlier = make_donation(
        donation_id="earlier", occurred=at(2026, 3, 1), recorded=at(2026, 3, 1)
    )
    store = InMemoryDonationStore([late_arrival, earlier])
    views = {d.donation_id: view for d, view in store.replay()}
    # Scored as of its own occurrence date in January, the late arrival cannot
    # see a donation that happened in March.
    assert len(views["late"]) == 0


def test_empty_view_is_usable():
    view = PointInTimeView((), at(2026, 1, 1))
    assert len(view) == 0
    assert view.by_sender("nobody") == ()
    assert view.sender_first_seen("nobody") is None
