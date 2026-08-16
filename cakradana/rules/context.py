"""Evaluation context and the limit table.

The limit table is derived from the rule set rather than declared separately.
The single-transaction limit rules already carry each threshold and its
citation, and the cumulative rules test the same thresholds against a sum. If
those numbers were written down twice they would eventually disagree, and a
disagreement between the limit used to test one donation and the limit used to
test that donation's running total is not something an analyst could ever
diagnose from the output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Iterable, Mapping

from cakradana.calendar import CampaignPeriod, ElectoralCalendar
from cakradana.history import PointInTimeView
from cakradana.registers import RegisterSet
from cakradana.schema import Donation, Entity, EntityRef
from cakradana.schema.enums import NON_INDIVIDUAL_DONOR_TYPES, EntityType, Regime
from cakradana.rules.schema import Rule, RuleSet


@dataclass(frozen=True)
class PeriodWindow:
    """The span a cumulative limit is measured over."""

    start: datetime
    end: datetime
    label: str

    def contains(self, when: datetime) -> bool:
        return self.start <= when <= self.end


@dataclass(frozen=True)
class Limit:
    """An applicable statutory limit and where it comes from."""

    amount_idr: int
    rule_id: str
    statute: str
    article: str
    regime: Regime


class LimitTable:
    """Applicable limits, indexed by regime and donor class."""

    def __init__(self, limits: dict[tuple[Regime, bool], Limit]) -> None:
        # Keyed by (regime, donor_is_individual). Donor class is reduced to a
        # boolean because every regime in force draws the same line: a natural
        # person on one side, every kind of organisation on the other.
        self._limits = limits

    @classmethod
    def from_ruleset(cls, ruleset: RuleSet) -> LimitTable:
        limits: dict[tuple[Regime, bool], Limit] = {}
        for rule in ruleset.tier1:
            basis = rule.legal_basis
            if basis is None or basis.threshold_idr is None:
                continue
            regime = rule.applies_when.regime
            if regime is None or regime is Regime.INDETERMINATE:
                continue
            for sender_type in rule.applies_when.sender_type or (EntityType.UNKNOWN,):
                key = (regime, sender_type is EntityType.INDIVIDUAL)
                if key in limits:
                    continue
                limits[key] = Limit(
                    amount_idr=basis.threshold_idr,
                    rule_id=rule.id,
                    statute=basis.statute,
                    article=basis.article,
                    regime=regime,
                )
        return cls(limits)

    def for_donation(
        self, regime: Regime, sender_type: EntityType
    ) -> Limit | None:
        if regime is Regime.INDETERMINATE:
            return None
        # An unknown donor type cannot be placed on either side of the
        # individual/organisation line. Falling through to the organisation
        # limit would be the wrong kind of wrong: it is the more permissive of
        # the two, so an unidentified donor would be judged against the highest
        # cap available and a breach of the individual limit would go unfound.
        if sender_type is EntityType.UNKNOWN:
            return None
        if sender_type is EntityType.INDIVIDUAL:
            return self._limits.get((regime, True))
        if sender_type in NON_INDIVIDUAL_DONOR_TYPES:
            return self._limits.get((regime, False))
        return None

    def __len__(self) -> int:
        return len(self._limits)


@dataclass
class RuleContext:
    """Everything a rule may consult about one donation.

    Rules are pure functions of this context. Given the same donation, the same
    history, and the same reference data, a rule always produces the same
    result, which is what makes a historical score reconstructible.
    """

    donation: Donation
    view: PointInTimeView
    calendar: ElectoralCalendar
    registers: RegisterSet
    limits: LimitTable
    now: datetime
    #: Resolved entity records, by id. Attributes that belong to a party rather
    #: than to a transaction — jurisdiction above all — are read from here, so
    #: that a rule cannot reach for them on the donation and quietly find
    #: nothing.
    entities: Mapping[str, Entity] = field(default_factory=dict)
    #: Annual periods are calendar years unless a deployment states otherwise.
    #: The statute fixes an annual cap without fixing the year boundary, so this
    #: is configuration rather than a constant.
    annual_period_start_month: int = 1

    def entity(self, ref: EntityRef) -> Entity | None:
        return self.entities.get(ref.entity_id) if ref.entity_id else None

    def sender_entity(self) -> Entity | None:
        return self.entity(self.donation.sender_ref)

    @property
    def regime(self) -> Regime:
        return self.calendar.regime_for(
            receiver_type=self.donation.receiver_ref.entity_type,
            electoral_context=self.donation.electoral_context,
            when=self.donation.occurred_at,
        )

    @property
    def applicable_limit(self) -> Limit | None:
        return self.limits.for_donation(
            self.regime, self.donation.sender_ref.entity_type
        )

    @property
    def campaign_period(self) -> CampaignPeriod | None:
        return self.calendar.period_for(
            self.donation.electoral_context, self.donation.occurred_at
        )

    def period_window(self) -> PeriodWindow | None:
        """The window a cumulative limit is measured over, if determinable."""
        regime = self.regime
        occurred = self.donation.occurred_at
        tz = occurred.tzinfo

        if regime is Regime.CAMPAIGN:
            period = self.campaign_period
            if period is None:
                return None
            return PeriodWindow(
                start=datetime.combine(period.start, datetime.min.time(), tzinfo=tz),
                end=datetime.combine(period.end, datetime.max.time(), tzinfo=tz),
                label=period.label or period.electoral_context,
            )

        if regime is Regime.PARTY_ANNUAL:
            start_year = occurred.year
            if occurred.month < self.annual_period_start_month:
                start_year -= 1
            start = datetime(
                start_year, self.annual_period_start_month, 1, tzinfo=tz
            )
            end = _add_year(start) - timedelta(microseconds=1)
            label = (
                str(start_year)
                if self.annual_period_start_month == 1
                else f"{start.date()}–{end.date()}"
            )
            return PeriodWindow(start=start, end=end, label=label)

        return None

    def missing_fields(self, required: Iterable[str]) -> tuple[str, ...]:
        """Which of a rule's declared inputs are unavailable on this donation."""
        missing: list[str] = []
        for field in required:
            if not self._has_field(field):
                missing.append(field)
        return tuple(missing)

    def _has_field(self, field: str) -> bool:
        donation = self.donation
        match field:
            case "amount":
                return donation.amount_idr > 0
            case "date":
                return donation.occurred_at is not None
            case "sender_type":
                return donation.sender_ref.entity_type is not EntityType.UNKNOWN
            case "receiver_type":
                return donation.receiver_ref.entity_type is not EntityType.UNKNOWN
            case "resolved_sender_id":
                return donation.sender_ref.is_resolved
            case "resolved_receiver_id":
                return donation.receiver_ref.is_resolved
            case "sender_jurisdiction":
                return False
            case "electoral_context":
                return donation.electoral_context is not None
            case "is_self_funded_declared":
                return donation.is_self_funded_declared is not None
            case "transaction_kind":
                return donation.transaction_kind is not None
            case _:
                return getattr(donation, field, None) is not None


def _add_year(when: datetime) -> datetime:
    try:
        return when.replace(year=when.year + 1)
    except ValueError:  # 29 February
        return when.replace(year=when.year + 1, day=28)


def rule_applies(rule: Rule, ctx: RuleContext) -> bool:
    return rule.applies_when.applies(
        ctx.donation.sender_ref.entity_type,
        ctx.donation.receiver_ref.entity_type,
        ctx.regime,
    )
