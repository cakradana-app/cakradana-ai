"""Group-level structural alerts.

A finding whose subject is a *set* of donations rather than one of them.

This is the difference between the graph lane being useful and being noise. A
smurfing cluster of forty donations, represented as forty independent
per-donation scores, is forty mediocre alerts that individually justify no
action — and an analyst who dismisses the first three never sees the pattern.
Represented as one alert naming the recipient, the donor set, and the window, it
is a single item that states what was observed.

Everything here is point-in-time. A cluster is assembled from what was knowable
at the moment it is assembled for, because a pattern detected using donations
recorded later is not a pattern that was detectable then, and scoring history
with it reproduces the leakage that invalidated the earlier metrics in a form
that is much harder to notice.
"""

from __future__ import annotations

import hashlib
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from cakradana.history import PointInTimeView
from cakradana.schema import Donation


class AlertKind(str, Enum):
    """What shape was observed. Never what it means."""

    FAN_IN_BURST = "FAN_IN_BURST"
    FAN_OUT = "FAN_OUT"
    LAYERING_CHAIN = "LAYERING_CHAIN"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


#: Which typology each shape is consistent with. Consistent with, not evidence
#: of: legitimate grassroots fundraising produces the same shape as smurfing,
#: and separating them is what the signals are for.
TYPOLOGY_OF: dict[AlertKind, str] = {
    AlertKind.FAN_IN_BURST: "T-09",
    AlertKind.FAN_OUT: "T-10",
    AlertKind.LAYERING_CHAIN: "T-10",
}


class AlertWindow(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    from_: datetime = Field(alias="from")
    to: datetime


class AlertSubject(BaseModel):
    """Who and what the alert is about.

    Donations are the subject. The entities are named because an analyst cannot
    act on a set of transaction identifiers, but the alert is a statement about
    a set of payments and not about the people party to them.
    """

    model_config = ConfigDict(frozen=True)

    #: The entity the shape converges on or radiates from.
    focus: str
    focus_role: str
    donations: tuple[str, ...]
    counterparties: tuple[str, ...]
    window: AlertWindow


class GroupAlert(BaseModel):
    """A structural finding about a set of donations."""

    model_config = ConfigDict(frozen=True)

    alert_id: str
    kind: AlertKind
    typology: str
    subject: AlertSubject
    #: The measured quantities the alert rests on, so a reader can disagree with
    #: the conclusion while still seeing the evidence.
    signals: dict[str, float | int | None]
    comparison: str | None
    score: int = Field(ge=0, le=100)
    #: How much of this pattern rests on donations whose party could not be
    #: resolved. A fan-in of twenty unresolved name variants may be one donor
    #: with inconsistent spelling rather than twenty donors, and an analyst
    #: needs that visible before acting.
    provisional_node_ratio: float = Field(ge=0.0, le=1.0)

    def covers(self, donation_id: str) -> bool:
        return donation_id in self.subject.donations


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

#: Window over which convergence is assessed. Long enough that splitting a
#: contribution across a few days still lands inside it, short enough that a
#: recipient's ordinary annual donor base does not.
DEFAULT_WINDOW_DAYS = 14

#: Distinct donors below which convergence is not remarkable. A recipient
#: drawing five donors in a fortnight is a recipient, not a pattern.
MIN_DISTINCT_DONORS = 8

#: Distinct recipients below which fan-out is not remarkable.
MIN_DISTINCT_RECIPIENTS = 6

#: Amounts within this fraction below a limit count as clustered against it.
#: Splitting to stay under a cap produces amounts that sit near it; ordinary
#: giving has no reason to.
THRESHOLD_BAND = 0.15

#: Chain search bounds. Layering is combinatorially expensive and an unbounded
#: search over a graph containing political parties never terminates usefully.
MAX_CHAIN_DEPTH = 4
CHAIN_EXPANSION_BUDGET = 2_000

#: A pass-through leg forwards most of what arrived, soon after it arrived.
CHAIN_MIN_FORWARD_RATIO = 0.6
CHAIN_MAX_LAG_DAYS = 14


@dataclass(frozen=True)
class DetectorSettings:
    window_days: int = DEFAULT_WINDOW_DAYS
    min_distinct_donors: int = MIN_DISTINCT_DONORS
    min_distinct_recipients: int = MIN_DISTINCT_RECIPIENTS
    #: The statutory limit amounts should be compared against, where one
    #: applies. Absent it, threshold proximity is reported as unmeasured rather
    #: than as zero — the two are different claims.
    threshold_idr: int | None = None
    max_chain_depth: int = MAX_CHAIN_DEPTH
    expansion_budget: int = CHAIN_EXPANSION_BUDGET


def _alert_id(kind: AlertKind, focus: str, donation_ids: Iterable[str]) -> str:
    """A stable identity for a cluster.

    Derived from its contents, so re-running detection over unchanged data
    produces the same alert rather than a second copy of it. An analyst who
    dispositioned a cluster yesterday should not be shown it again today under
    a new identifier.
    """
    digest = hashlib.sha256()
    digest.update(kind.value.encode())
    digest.update(b"|")
    digest.update(focus.encode())
    for donation_id in sorted(donation_ids):
        digest.update(b"|")
        digest.update(donation_id.encode())
    return f"cluster:{digest.hexdigest()[:16]}"


def _coefficient_of_variation(values: Sequence[float]) -> float | None:
    """How uniform a set of amounts is.

    Organic giving is heterogeneous: people give what they can. A set of
    near-identical amounts arriving together is the signature of one decision
    executed many times.
    """
    if len(values) < 2:
        return None
    mean = statistics.fmean(values)
    if mean == 0:
        return None
    return statistics.pstdev(values) / mean


class GroupAlertDetector:
    """Finds structural patterns over a point-in-time view."""

    def __init__(self, settings: DetectorSettings | None = None) -> None:
        self.settings = settings or DetectorSettings()

    def detect(
        self, view: PointInTimeView, *, as_of: datetime | None = None
    ) -> tuple[GroupAlert, ...]:
        end = as_of or view.as_of
        start = end - timedelta(days=self.settings.window_days)
        window = AlertWindow(**{"from": start, "to": end})
        recent = [d for d in view if start <= d.occurred_at <= end]

        alerts = [
            *self._fan_in(view, recent, window),
            *self._fan_out(recent, window),
            *self._chains(recent, window),
        ]
        # Highest first: the queue this feeds is bounded, and an analyst reading
        # from the top should meet the strongest structure first.
        return tuple(sorted(alerts, key=lambda a: a.score, reverse=True))

    # -- fan-in ---------------------------------------------------------------

    def _fan_in(
        self,
        view: PointInTimeView,
        recent: Sequence[Donation],
        window: AlertWindow,
    ) -> list[GroupAlert]:
        by_receiver: dict[str, list[Donation]] = defaultdict(list)
        for donation in recent:
            if donation.receiver_ref.is_resolved:
                by_receiver[donation.receiver_ref.key].append(donation)

        baseline = self._median_fan_in(by_receiver)
        alerts = []
        for receiver, donations in by_receiver.items():
            donors = {
                d.sender_ref.key for d in donations if d.sender_ref.is_resolved
            }
            if len(donors) < self.settings.min_distinct_donors:
                continue

            amounts = [float(d.amount_idr) for d in donations]
            cv = _coefficient_of_variation(amounts)
            # Donors with no history before this window opened. A base that
            # appeared all at once is a different thing from one that grew.
            thin = sum(
                1
                for donor in donors
                if not view.has_prior_history(donor, before=window.from_)
            )
            provisional = sum(
                1 for d in donations if not d.sender_ref.is_resolved
            ) / len(donations)

            signals: dict[str, float | int | None] = {
                "distinct_donors": len(donors),
                "donations": len(donations),
                "donor_thinness_ratio": round(thin / len(donors), 4),
                "amount_cv": None if cv is None else round(cv, 4),
                "threshold_proximity_ratio": self._threshold_proximity(amounts),
                "total_idr": int(sum(amounts)),
            }
            score = self._fan_in_score(signals, len(donors), baseline, provisional)
            alerts.append(
                GroupAlert(
                    alert_id=_alert_id(
                        AlertKind.FAN_IN_BURST,
                        receiver,
                        (d.donation_id for d in donations),
                    ),
                    kind=AlertKind.FAN_IN_BURST,
                    typology=TYPOLOGY_OF[AlertKind.FAN_IN_BURST],
                    subject=AlertSubject(
                        focus=receiver,
                        focus_role="recipient",
                        donations=tuple(sorted(d.donation_id for d in donations)),
                        counterparties=tuple(sorted(donors)),
                        window=window,
                    ),
                    signals=signals,
                    comparison=(
                        f"The median recipient drew {baseline} distinct donor(s) "
                        f"in this window."
                    ),
                    score=score,
                    provisional_node_ratio=round(provisional, 4),
                )
            )
        return alerts

    @staticmethod
    def _median_fan_in(by_receiver: dict[str, list[Donation]]) -> int:
        """The ordinary case, measured rather than assumed.

        A signal without a reference point is not actionable. Twenty-three
        donors means nothing until it sits beside what a recipient usually
        draws in the same period.
        """
        counts = [
            len({d.sender_ref.key for d in donations if d.sender_ref.is_resolved})
            for donations in by_receiver.values()
        ]
        return int(statistics.median(counts)) if counts else 0

    def _threshold_proximity(self, amounts: Sequence[float]) -> float | None:
        threshold = self.settings.threshold_idr
        if not threshold:
            # Not measured, and reported as such. Returning zero would assert
            # that no amount clusters below the limit, which is a finding this
            # detector has no basis for.
            return None
        floor = threshold * (1 - THRESHOLD_BAND)
        near = sum(1 for amount in amounts if floor <= amount < threshold)
        return round(near / len(amounts), 4) if amounts else None

    @staticmethod
    def _fan_in_score(
        signals: dict[str, float | int | None],
        donors: int,
        baseline: int,
        provisional: float,
    ) -> int:
        """How strongly the shape departs from ordinary fundraising.

        Weighted toward the signals that separate splitting from a genuine
        grassroots surge: donors who exist only for this, amounts that are all
        the same, amounts that sit just below a cap. Volume alone is the
        weakest of the four, because a popular candidate produces it honestly.
        """
        excess = min(donors / max(baseline * 4, 8), 1.0)
        thinness = signals["donor_thinness_ratio"] or 0.0
        cv = signals["amount_cv"]
        homogeneity = 0.0 if cv is None else max(0.0, 1.0 - min(cv / 0.5, 1.0))
        proximity = signals["threshold_proximity_ratio"] or 0.0

        intensity = (
            0.20 * excess + 0.30 * thinness + 0.30 * homogeneity + 0.20 * proximity
        )
        # Downweighted by how much rests on unresolved parties, per `15` §2.
        return round(100 * intensity * (1 - provisional))

    # -- fan-out --------------------------------------------------------------

    def _fan_out(
        self, recent: Sequence[Donation], window: AlertWindow
    ) -> list[GroupAlert]:
        by_sender: dict[str, list[Donation]] = defaultdict(list)
        for donation in recent:
            if donation.sender_ref.is_resolved:
                by_sender[donation.sender_ref.key].append(donation)

        alerts = []
        for sender, donations in by_sender.items():
            recipients = {
                d.receiver_ref.key for d in donations if d.receiver_ref.is_resolved
            }
            if len(recipients) < self.settings.min_distinct_recipients:
                continue

            amounts = [float(d.amount_idr) for d in donations]
            cv = _coefficient_of_variation(amounts)
            provisional = sum(
                1 for d in donations if not d.receiver_ref.is_resolved
            ) / len(donations)

            signals: dict[str, float | int | None] = {
                "distinct_recipients": len(recipients),
                "donations": len(donations),
                # Near-equal outflows suggest a total being partitioned rather
                # than a series of independent decisions.
                "amount_partition_regularity": (
                    None if cv is None else round(max(0.0, 1.0 - min(cv / 0.5, 1.0)), 4)
                ),
                "total_idr": int(sum(amounts)),
            }
            regularity = signals["amount_partition_regularity"] or 0.0
            breadth = min(len(recipients) / 12.0, 1.0)
            score = round(100 * (0.5 * breadth + 0.5 * regularity) * (1 - provisional))

            alerts.append(
                GroupAlert(
                    alert_id=_alert_id(
                        AlertKind.FAN_OUT, sender, (d.donation_id for d in donations)
                    ),
                    kind=AlertKind.FAN_OUT,
                    typology=TYPOLOGY_OF[AlertKind.FAN_OUT],
                    subject=AlertSubject(
                        focus=sender,
                        focus_role="donor",
                        donations=tuple(sorted(d.donation_id for d in donations)),
                        counterparties=tuple(sorted(recipients)),
                        window=window,
                    ),
                    signals=signals,
                    comparison="Most donors give to one or two recipients.",
                    score=score,
                    provisional_node_ratio=round(provisional, 4),
                )
            )
        return alerts

    # -- layering chains ------------------------------------------------------

    def _chains(
        self, recent: Sequence[Donation], window: AlertWindow
    ) -> list[GroupAlert]:
        """Funds traversing several entities before reaching a recipient.

        Bounded by depth and by an expansion budget. Both bounds are reported
        on the alert rather than applied silently: a search that stopped early
        found what it found, and saying so is the difference between a bounded
        result and an apparently complete one.
        """
        outgoing: dict[str, list[Donation]] = defaultdict(list)
        for donation in recent:
            if donation.sender_ref.is_resolved and donation.receiver_ref.is_resolved:
                outgoing[donation.sender_ref.key].append(donation)

        alerts: list[GroupAlert] = []
        expansions = 0
        seen_paths: set[tuple[str, ...]] = set()

        for start_node, first_legs in outgoing.items():
            for first in first_legs:
                stack: list[tuple[list[Donation], set[str]]] = [
                    ([first], {start_node, first.receiver_ref.key})
                ]
                while stack:
                    if expansions >= self.settings.expansion_budget:
                        break
                    path, visited = stack.pop()
                    expansions += 1
                    head = path[-1]
                    for nxt in outgoing.get(head.receiver_ref.key, ()):
                        if nxt.receiver_ref.key in visited:
                            continue
                        if not self._forwards(head, nxt):
                            continue
                        extended = [*path, nxt]
                        if len(extended) >= 3:
                            key = tuple(d.donation_id for d in extended)
                            if key not in seen_paths:
                                seen_paths.add(key)
                                alerts.append(self._chain_alert(extended, window))
                        if len(extended) < self.settings.max_chain_depth:
                            stack.append(
                                (extended, visited | {nxt.receiver_ref.key})
                            )
        return alerts

    @staticmethod
    def _forwards(inflow: Donation, outflow: Donation) -> bool:
        """Whether one donation plausibly carries the other onward.

        Most of what arrived, moving on shortly after it arrived. An entity
        that receives and then gives an unrelated amount months later is not a
        conduit; it is an entity that does two things.
        """
        if outflow.occurred_at < inflow.occurred_at:
            return False
        lag_days = (outflow.occurred_at - inflow.occurred_at).total_seconds() / 86400
        if lag_days > CHAIN_MAX_LAG_DAYS:
            return False
        return outflow.amount_idr >= inflow.amount_idr * CHAIN_MIN_FORWARD_RATIO

    def _chain_alert(
        self, path: Sequence[Donation], window: AlertWindow
    ) -> GroupAlert:
        first, last = path[0], path[-1]
        lag = (last.occurred_at - first.occurred_at).total_seconds() / 86400
        attenuation = last.amount_idr / first.amount_idr if first.amount_idr else None
        hops = len(path)
        intermediaries = tuple(d.receiver_ref.key for d in path[:-1])

        signals: dict[str, float | int | None] = {
            "hops": hops,
            "total_lag_days": round(lag, 2),
            "amount_attenuation": None if attenuation is None else round(attenuation, 4),
            "origin_idr": first.amount_idr,
            "terminal_idr": last.amount_idr,
        }
        # Longer chains that lose little along the way are the ones worth a
        # look. A chain is weak evidence on its own — these are ordinary
        # donations until something explains why they line up.
        depth = min((hops - 2) / 2.0, 1.0)
        retention = min(attenuation or 0.0, 1.0)
        score = round(100 * (0.6 * depth + 0.4 * retention) * 0.7)

        return GroupAlert(
            alert_id=_alert_id(
                AlertKind.LAYERING_CHAIN,
                first.sender_ref.key,
                (d.donation_id for d in path),
            ),
            kind=AlertKind.LAYERING_CHAIN,
            typology=TYPOLOGY_OF[AlertKind.LAYERING_CHAIN],
            subject=AlertSubject(
                focus=first.sender_ref.key,
                focus_role="origin",
                donations=tuple(d.donation_id for d in path),
                counterparties=intermediaries,
                window=window,
            ),
            signals=signals,
            comparison=(
                "Funds ordinarily reach a recipient in one hop; this reached it "
                f"in {hops}."
            ),
            score=score,
            provisional_node_ratio=0.0,
        )


class AlertIndex:
    """Group alerts, addressable by the donations they cover.

    Built once per detection pass and consulted per donation, because the
    alternative — re-detecting for each donation being scored — makes the cost
    of the lane quadratic in the size of the cluster it is describing.
    """

    def __init__(self, alerts: Iterable[GroupAlert] = ()) -> None:
        self.alerts: tuple[GroupAlert, ...] = tuple(alerts)
        self._by_donation: dict[str, list[GroupAlert]] = defaultdict(list)
        for alert in self.alerts:
            for donation_id in alert.subject.donations:
                self._by_donation[donation_id].append(alert)

    def covering(self, donation_id: str) -> tuple[GroupAlert, ...]:
        return tuple(self._by_donation.get(donation_id, ()))

    def __len__(self) -> int:
        return len(self.alerts)

    def __iter__(self):
        return iter(self.alerts)
