"""Shared fixtures.

``WIB`` is used rather than UTC throughout the tests because every statutory
period boundary in this domain is expressed in local time, and a test suite
that only ever exercises UTC would not catch an off-by-one at a period edge.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from cakradana.schema import Channel, Donation, EntityRef, EntityType

WIB = timezone(timedelta(hours=7), name="WIB")


def at(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=WIB)


def make_donation(
    donation_id: str = "d-1",
    sender: str = "e-sender",
    receiver: str = "e-receiver",
    amount_idr: int = 10_000_000,
    occurred: datetime | None = None,
    recorded: datetime | None = None,
    sender_type: EntityType = EntityType.INDIVIDUAL,
    receiver_type: EntityType = EntityType.POLITICAL_PARTY,
    **kwargs,
) -> Donation:
    occurred = occurred or at(2026, 6, 1)
    recorded = recorded if recorded is not None else occurred
    return Donation(
        donation_id=donation_id,
        sender_ref=EntityRef(entity_id=sender, entity_type=sender_type),
        receiver_ref=EntityRef(entity_id=receiver, entity_type=receiver_type),
        amount_idr=amount_idr,
        occurred_at=occurred,
        recorded_at=recorded,
        channel=kwargs.pop("channel", Channel.DIGITAL_FORM),
        **kwargs,
    )


@pytest.fixture
def donation() -> Donation:
    return make_donation()
