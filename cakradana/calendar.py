"""Electoral periods and limit-regime selection.

Two statutory limit regimes coexist. A donation to a political party outside a
campaign period falls under the annual party regime; the same donor giving to
an election participant during a campaign falls under the campaign regime, and
the campaign limits are an order of magnitude higher.

Applying one regime universally produces both false findings and missed ones,
so regime selection is part of evaluating every limit rule.

Where the campaign period for an electoral context is not known, selection
returns indeterminate. It never falls back to the more permissive regime: doing
so would convert missing configuration into a silent clean result, which is the
failure mode the whole engine is built to avoid.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Iterable

from pydantic import BaseModel, ConfigDict, model_validator

from cakradana.schema.enums import EntityType, Regime


class CampaignPeriod(BaseModel):
    """A declared campaign period for one electoral context."""

    model_config = ConfigDict(frozen=True)

    electoral_context: str
    start: date
    end: date
    #: Reporting deadlines within the period, used to detect donations bunched
    #: immediately before a filing cut-off.
    reporting_deadlines: tuple[date, ...] = ()
    label: str | None = None

    @model_validator(mode="after")
    def _ordered(self) -> CampaignPeriod:
        if self.end < self.start:
            raise ValueError(
                f"{self.electoral_context}: campaign period ends before it starts"
            )
        return self

    def covers(self, when: date) -> bool:
        return self.start <= when <= self.end

    def next_deadline_after(self, when: date) -> date | None:
        upcoming = sorted(d for d in self.reporting_deadlines if d >= when)
        return upcoming[0] if upcoming else None


class ElectoralCalendar:
    """Known campaign periods, keyed by electoral context.

    An empty calendar is a valid and deliberately conservative state: with no
    periods configured, every campaign-regime rule reports indeterminate rather
    than guessing.
    """

    def __init__(self, periods: Iterable[CampaignPeriod] = ()) -> None:
        self._periods: dict[str, tuple[CampaignPeriod, ...]] = {}
        for period in periods:
            existing = self._periods.get(period.electoral_context, ())
            self._periods[period.electoral_context] = existing + (period,)

    def __len__(self) -> int:
        return sum(len(v) for v in self._periods.values())

    def knows(self, electoral_context: str | None) -> bool:
        return electoral_context is not None and electoral_context in self._periods

    def period_for(
        self, electoral_context: str | None, when: datetime | date
    ) -> CampaignPeriod | None:
        """The campaign period covering ``when``, if one is configured."""
        if electoral_context is None:
            return None
        day = when.date() if isinstance(when, datetime) else when
        for period in self._periods.get(electoral_context, ()):
            if period.covers(day):
                return period
        return None

    def regime_for(
        self,
        *,
        receiver_type: EntityType,
        electoral_context: str | None,
        when: datetime | date,
    ) -> Regime:
        """Select the limit regime governing a donation.

        A donation reaching a recipient inside a declared campaign period is
        governed by the campaign limits. A donation to a political party whose
        date can be positively placed outside every declared campaign period is
        governed by the annual party limits.

        Everything else is indeterminate, including a donation whose electoral
        context is absent or not configured. Assuming the annual regime in that
        case would be the stricter choice numerically, but strictness is not
        the objective: the annual limit is an order of magnitude below the
        campaign limit, so a lawful campaign donation judged against it yields
        a false statutory finding. Statutory findings are asserted as fact and
        must be right, so an unplaceable date produces no finding at all.

        The cost is that a deployment with no calendar configured evaluates no
        limit rules. That cost is deliberately visible: the share of donations
        returning indeterminate is an operational metric, so an unconfigured
        calendar shows up as unevaluated rather than as clean results.
        """
        if self.period_for(electoral_context, when) is not None:
            return Regime.CAMPAIGN

        if receiver_type is EntityType.POLITICAL_PARTY and self.knows(
            electoral_context
        ):
            return Regime.PARTY_ANNUAL

        return Regime.INDETERMINATE
