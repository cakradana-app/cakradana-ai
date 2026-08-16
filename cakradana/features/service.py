"""Feature computation.

Two paths, one set of definitions. Training replays donations in the order the
system learned of them and snapshots the feature vector at each point; serving
computes the same vector on request. Both call the same functions, so the only
way for them to disagree is for the underlying history to disagree.

Every scoring event keeps the exact vector it used. That is what makes a score
reconstructible later, lets an explanation be checked against what the model
actually saw, and lets training reuse the values that were served rather than
recomputing them under assumptions that have since changed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator, Mapping, Sequence

from cakradana.calendar import ElectoralCalendar
from cakradana.features.definitions import (
    FeatureValue,
    catalogue,
    categorical_names,
    feature_names,
    numeric_names,
)
from cakradana.history import DonationStore, PointInTimeView
from cakradana.registers import RegisterSet, empty_register_set
from cakradana.rules.context import LimitTable, RuleContext
from cakradana.rules.schema import RuleSet
from cakradana.schema import Donation, Entity


@dataclass(frozen=True)
class FeatureVector:
    """One donation's features, with the definition version that produced them."""

    donation_id: str
    donation_version: int
    computed_at: datetime
    feature_set_version: str
    values: Mapping[str, FeatureValue]

    def as_row(self, order: Sequence[str]) -> list[FeatureValue]:
        return [self.values.get(name) for name in order]

    def missing(self) -> tuple[str, ...]:
        return tuple(k for k, v in self.values.items() if v is None)


class FeatureService:
    """Computes feature vectors for donations.

    The rule engine and this service share one context type deliberately. If
    each built its own view of a donation's history they could disagree about
    what was knowable, and a finding would then cite a total the features
    contradict.
    """

    def __init__(
        self,
        ruleset: RuleSet,
        *,
        calendar: ElectoralCalendar | None = None,
        registers: RegisterSet | None = None,
        annual_period_start_month: int = 1,
    ) -> None:
        self.calendar = calendar or ElectoralCalendar()
        self.registers = registers or empty_register_set()
        self.limits = LimitTable.from_ruleset(ruleset)
        self.annual_period_start_month = annual_period_start_month
        self._catalogue = catalogue()
        self.version = feature_set_version()

    @property
    def names(self) -> tuple[str, ...]:
        return feature_names()

    @property
    def categorical(self) -> tuple[str, ...]:
        return categorical_names()

    @property
    def numeric(self) -> tuple[str, ...]:
        return numeric_names()

    def context_for(
        self,
        donation: Donation,
        view: PointInTimeView,
        *,
        now: datetime | None = None,
        entities: Mapping[str, Entity] | None = None,
    ) -> RuleContext:
        return RuleContext(
            donation=donation,
            view=view,
            calendar=self.calendar,
            registers=self.registers,
            limits=self.limits,
            now=now or donation.occurred_at,
            entities=entities or {},
            annual_period_start_month=self.annual_period_start_month,
        )

    def compute(
        self,
        donation: Donation,
        view: PointInTimeView,
        *,
        now: datetime | None = None,
        entities: Mapping[str, Entity] | None = None,
    ) -> FeatureVector:
        ctx = self.context_for(donation, view, now=now, entities=entities)
        return self.compute_from_context(ctx)

    def compute_from_context(self, ctx: RuleContext) -> FeatureVector:
        values: dict[str, FeatureValue] = {}
        for name, spec in self._catalogue.items():
            values[name] = spec.compute(ctx)
        return FeatureVector(
            donation_id=ctx.donation.donation_id,
            donation_version=ctx.donation.donation_version,
            computed_at=ctx.now,
            feature_set_version=self.version,
            values=values,
        )

    def backfill(
        self, store: DonationStore, *, entities: Mapping[str, Entity] | None = None
    ) -> Iterator[tuple[Donation, FeatureVector]]:
        """Replay a store, yielding the vector each donation would have had.

        Ordered by when the system learned of each donation, so that the
        sequence of states matches the one serving would have passed through.
        Computing features over a completed dataset instead — the shape of the
        earlier pipeline — lets every row see the whole history and produces
        measurements that cannot be reproduced at serving time.
        """
        for donation, view in store.replay():
            yield donation, self.compute(donation, view, entities=entities)


def feature_set_version() -> str:
    """A fingerprint of the active feature definitions.

    Derived from the names and declared types rather than set by hand, so a
    definition cannot change without the version recorded on every score
    changing with it.
    """
    spec = [
        {"name": s.name, "family": s.family, "dtype": s.dtype}
        for s in catalogue().values()
    ]
    digest = hashlib.sha256(
        json.dumps(spec, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return f"features-{digest[:12]}"
