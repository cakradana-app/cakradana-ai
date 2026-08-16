"""Reference registers.

Some statutory prohibitions are lookups, not inferences. Whether a donor is a
state-owned enterprise is a question of fact answered by an authoritative list,
and the system either has that list or it does not.

Name patterns may nominate candidates for a register. They may never produce a
finding. "PT Sumber Sejahtera" is an ordinary company and "PT PLN (Persero)" is
a state enterprise, and no prefix rule separates them; a system that guessed
would accuse real companies of an offence on the strength of their name.

An unavailable register therefore yields indeterminate, never a pass. A stale
one does the same once past its freshness horizon, because a register that has
silently stopped being updated degrades a prohibition into a rubber stamp.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Iterable

from pydantic import BaseModel, ConfigDict


class RegisterEntry(BaseModel):
    """One entity's membership of a register, with the dates it applied."""

    model_config = ConfigDict(frozen=True)

    entity_id: str
    canonical_name: str
    aliases: tuple[str, ...] = ()
    category: str | None = None
    effective_from: date | None = None
    effective_to: date | None = None
    source: str | None = None

    def covers(self, when: date) -> bool:
        if self.effective_from and when < self.effective_from:
            return False
        if self.effective_to and when > self.effective_to:
            return False
        return True


class RegisterLookup(BaseModel):
    """Outcome of a register query.

    ``available`` and ``member`` are separate because "this donor is not on the
    prohibited list" and "there is no prohibited list" are entirely different
    statements, and only the first is evidence of anything.
    """

    model_config = ConfigDict(frozen=True)

    register_name: str
    available: bool
    member: bool = False
    entry: RegisterEntry | None = None
    reason: str | None = None


class Register:
    """A named reference list with freshness metadata.

    Constructed unavailable by default. The lists this system needs — state
    enterprises, government bodies, final criminal convictions — are held by
    other institutions, and a register is only available once one of them has
    actually supplied it.
    """

    def __init__(
        self,
        name: str,
        entries: Iterable[RegisterEntry] = (),
        *,
        available: bool = False,
        refreshed_at: datetime | None = None,
        max_age: timedelta | None = None,
        unavailable_reason: str = "register not supplied",
    ) -> None:
        self.name = name
        self._entries = {e.entity_id: e for e in entries}
        self._by_name: dict[str, RegisterEntry] = {}
        for entry in self._entries.values():
            self._by_name[_normalise(entry.canonical_name)] = entry
            for alias in entry.aliases:
                self._by_name[_normalise(alias)] = entry
        self._available = available
        self._refreshed_at = refreshed_at
        self._max_age = max_age
        self._unavailable_reason = unavailable_reason

    def __len__(self) -> int:
        return len(self._entries)

    def is_stale(self, now: datetime) -> bool:
        if self._max_age is None:
            return False
        if self._refreshed_at is None:
            return True
        return now - self._refreshed_at > self._max_age

    def lookup(
        self,
        entity_id: str | None,
        *,
        when: date,
        now: datetime | None = None,
        name: str | None = None,
    ) -> RegisterLookup:
        if not self._available:
            return RegisterLookup(
                register_name=self.name,
                available=False,
                reason=self._unavailable_reason,
            )

        now = now or datetime.now(tz=timezone.utc)
        if self._max_age is not None and self.is_stale(now):
            return RegisterLookup(
                register_name=self.name,
                available=False,
                reason="register is stale and cannot be relied on",
            )

        entry = self._entries.get(entity_id) if entity_id else None
        if entry is None and name:
            entry = self._by_name.get(_normalise(name))

        if entry is None:
            return RegisterLookup(register_name=self.name, available=True, member=False)
        if not entry.covers(when):
            return RegisterLookup(
                register_name=self.name,
                available=True,
                member=False,
                reason="entry exists but did not apply on this date",
            )
        return RegisterLookup(
            register_name=self.name, available=True, member=True, entry=entry
        )


def _normalise(value: str) -> str:
    """Fold a name for matching.

    Deliberately conservative: case and whitespace only. Aggressive
    normalisation raises the chance of matching an unrelated entity onto a
    prohibited-source entry, and the cost of that error is an accusation.
    """
    return " ".join(value.strip().casefold().split())


class RegisterSet:
    """The registers available to the rule engine."""

    #: Donations from these sources are prohibited outright by statute.
    PROHIBITED_SOURCE = "prohibited_source"
    #: Convictions with final legal force. Reporting on an investigation or a
    #: named suspect does not qualify, so adverse media can never populate it.
    FINAL_CONVICTIONS = "final_convictions"

    def __init__(self, registers: Iterable[Register] = ()) -> None:
        self._registers = {r.name: r for r in registers}

    def get(self, name: str) -> Register:
        return self._registers.get(name) or Register(
            name, unavailable_reason=f"register {name!r} is not configured"
        )

    def lookup(
        self,
        name: str,
        entity_id: str | None,
        *,
        when: date,
        now: datetime | None = None,
        entity_name: str | None = None,
    ) -> RegisterLookup:
        return self.get(name).lookup(entity_id, when=when, now=now, name=entity_name)


def empty_register_set() -> RegisterSet:
    """The default posture: every register declared, none supplied.

    Declaring them keeps the dependent rules loadable and keeps their absence
    reportable, rather than leaving the prohibitions unrepresented.
    """
    return RegisterSet(
        [
            Register(
                RegisterSet.PROHIBITED_SOURCE,
                unavailable_reason=(
                    "no authoritative register of government bodies, state and "
                    "regional enterprises, and village governments has been supplied"
                ),
            ),
            Register(
                RegisterSet.FINAL_CONVICTIONS,
                unavailable_reason=(
                    "no authoritative register of convictions with final legal "
                    "force has been supplied"
                ),
            ),
        ]
    )
