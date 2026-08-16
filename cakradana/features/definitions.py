"""Feature definitions.

One implementation of each feature, used identically by training and serving.
A second implementation "for inference" is the defect this module exists to
prevent: two implementations drift, and when they do the model is served inputs
whose distribution it never saw, with nothing reporting the failure.

Three rules govern what may appear here.

Aggregates look strictly backwards. A feature describing a donor's prior
behaviour never sees the donation being scored.

An uncomputable feature is null, never zero. A donor's first donation has no
standard deviation and no mean interval; filling those with zero states that
the donor is perfectly regular, which is the opposite of what is known. Nulls
reach the model as genuine missing values, which it handles natively.

Rule outputs are not features. A statutory verdict is a legal finding, not a
model input, and feeding a heuristic's own output back as a feature is exactly
how a classifier learns to reproduce the heuristic instead of improving on it.
Continuous quantities near a threshold are permitted and are a different thing:
the distance to a limit lets the model learn behaviour approaching one, whereas
the verdict would hand it the answer.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import timedelta
from typing import Callable, Literal

from cakradana.rules.context import RuleContext
from cakradana.schema.enums import Regime

FeatureValue = float | int | bool | str | None
Compute = Callable[[RuleContext], FeatureValue]

Family = Literal[
    "transaction",
    "donor",
    "recipient",
    "pair",
    "temporal",
    "graph",
    "threshold",
    "reputation",
    "quality",
]


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    family: Family
    dtype: Literal["int", "float", "bool", "categorical"]
    description: str
    compute: Compute
    #: When this feature is legitimately null. Stated so that a null in
    #: production can be told from a pipeline defect.
    null_when: str = "never"

    @property
    def is_categorical(self) -> bool:
        return self.dtype == "categorical"


_CATALOGUE: dict[str, FeatureSpec] = {}


def feature(
    name: str,
    family: Family,
    dtype: Literal["int", "float", "bool", "categorical"],
    description: str,
    null_when: str = "never",
) -> Callable[[Compute], Compute]:
    def register(fn: Compute) -> Compute:
        if name in _CATALOGUE:
            raise ValueError(f"duplicate feature {name!r}")
        _CATALOGUE[name] = FeatureSpec(
            name=name,
            family=family,
            dtype=dtype,
            description=description,
            compute=fn,
            null_when=null_when,
        )
        return fn

    return register


def catalogue() -> dict[str, FeatureSpec]:
    return dict(_CATALOGUE)


def feature_names() -> tuple[str, ...]:
    return tuple(_CATALOGUE)


def categorical_names() -> tuple[str, ...]:
    return tuple(n for n, s in _CATALOGUE.items() if s.is_categorical)


def numeric_names() -> tuple[str, ...]:
    return tuple(n for n, s in _CATALOGUE.items() if not s.is_categorical)


# ---------------------------------------------------------------------------
# Transaction-intrinsic — always available
# ---------------------------------------------------------------------------


@feature("amount", "transaction", "int", "Donation amount in rupiah")
def _amount(ctx: RuleContext) -> int:
    return ctx.donation.amount_idr


@feature("amount_log", "transaction", "float", "log1p of the amount")
def _amount_log(ctx: RuleContext) -> float:
    return math.log1p(ctx.donation.amount_idr)


@feature("sender_type", "transaction", "categorical", "Donor entity type")
def _sender_type(ctx: RuleContext) -> str:
    return str(ctx.donation.sender_ref.entity_type)


@feature("receiver_type", "transaction", "categorical", "Recipient entity type")
def _receiver_type(ctx: RuleContext) -> str:
    return str(ctx.donation.receiver_ref.entity_type)


@feature("transaction_kind", "transaction", "categorical", "How value moved")
def _transaction_kind(ctx: RuleContext) -> str:
    return str(ctx.donation.transaction_kind)


@feature("channel", "transaction", "categorical", "Ingestion channel")
def _channel(ctx: RuleContext) -> str:
    return str(ctx.donation.channel)


@feature(
    "is_round_amount",
    "transaction",
    "bool",
    "Amount is a round figure at a million rupiah or above",
)
def _is_round_amount(ctx: RuleContext) -> bool:
    """Coordinated donations tend to round numbers; organic ones less so."""
    return ctx.donation.amount_idr % 1_000_000 == 0


@feature(
    "amount_trailing_zeros", "transaction", "int", "Count of trailing zeros"
)
def _amount_trailing_zeros(ctx: RuleContext) -> int:
    digits = str(ctx.donation.amount_idr)
    return len(digits) - len(digits.rstrip("0"))


# ---------------------------------------------------------------------------
# Donor behavioural — strictly prior
# ---------------------------------------------------------------------------


def _prior_from_sender(ctx: RuleContext):
    sender = ctx.donation.sender_ref
    if not sender.is_resolved:
        return None
    return ctx.view.by_sender(
        sender.key,
        until=ctx.donation.occurred_at,
        excluding=ctx.donation.donation_id,
    )


@feature(
    "total_donasi_sender",
    "donor",
    "float",
    "Sum of the donor's prior donations",
    null_when="donor is unresolved",
)
def _total_donasi_sender(ctx: RuleContext) -> float | None:
    prior = _prior_from_sender(ctx)
    return None if prior is None else float(sum(d.amount_idr for d in prior))


@feature(
    "jumlah_transaksi_sender",
    "donor",
    "int",
    "Count of the donor's prior donations",
    null_when="donor is unresolved",
)
def _jumlah_transaksi_sender(ctx: RuleContext) -> int | None:
    prior = _prior_from_sender(ctx)
    return None if prior is None else len(prior)


@feature(
    "rata_rata_donasi_sender",
    "donor",
    "float",
    "Mean of the donor's prior donations",
    null_when="donor is unresolved or has no prior donations",
)
def _rata_rata_donasi_sender(ctx: RuleContext) -> float | None:
    prior = _prior_from_sender(ctx)
    if not prior:
        return None
    return statistics.fmean(d.amount_idr for d in prior)


@feature(
    "std_donasi_sender",
    "donor",
    "float",
    "Standard deviation of the donor's prior donations",
    null_when="donor has fewer than two prior donations",
)
def _std_donasi_sender(ctx: RuleContext) -> float | None:
    """Undefined before a donor has a spread to measure.

    Zero would state that the donor gives an unvarying amount every time,
    which is a strong signal and the opposite of what a single observation
    supports.
    """
    prior = _prior_from_sender(ctx)
    if prior is None or len(prior) < 2:
        return None
    return statistics.stdev(d.amount_idr for d in prior)


@feature(
    "jumlah_donasi_30hari_sender",
    "donor",
    "int",
    "Donor's donations in the preceding 30 days",
    null_when="donor is unresolved",
)
def _jumlah_donasi_30hari_sender(ctx: RuleContext) -> int | None:
    sender = ctx.donation.sender_ref
    if not sender.is_resolved:
        return None
    return len(
        ctx.view.by_sender(
            sender.key,
            since=ctx.donation.occurred_at - timedelta(days=30),
            until=ctx.donation.occurred_at,
            excluding=ctx.donation.donation_id,
        )
    )


@feature(
    "selang_waktu_rata2_sender",
    "donor",
    "float",
    "Mean interval in days between the donor's prior donations",
    null_when="donor has fewer than two prior donations",
)
def _selang_waktu_rata2_sender(ctx: RuleContext) -> float | None:
    prior = _prior_from_sender(ctx)
    if prior is None or len(prior) < 2:
        return None
    stamps = sorted(d.occurred_at for d in prior)
    gaps = [
        (later - earlier).total_seconds() / 86400
        for earlier, later in zip(stamps, stamps[1:])
    ]
    return statistics.fmean(gaps)


@feature(
    "receiver_unik_per_sender",
    "donor",
    "int",
    "Distinct recipients the donor has given to before",
    null_when="donor is unresolved",
)
def _receiver_unik_per_sender(ctx: RuleContext) -> int | None:
    prior = _prior_from_sender(ctx)
    if prior is None:
        return None
    return len({d.receiver_ref.entity_id for d in prior if d.receiver_ref.entity_id})


@feature(
    "max_donasi_satu_receiver",
    "donor",
    "float",
    "Largest cumulative amount the donor has given to any one recipient",
    null_when="donor is unresolved or has no prior donations",
)
def _max_donasi_satu_receiver(ctx: RuleContext) -> float | None:
    prior = _prior_from_sender(ctx)
    if not prior:
        return None
    totals: dict[str, int] = {}
    for d in prior:
        key = d.receiver_ref.entity_id
        if key:
            totals[key] = totals.get(key, 0) + d.amount_idr
    return float(max(totals.values())) if totals else None


@feature(
    "proporsi_donasi_terbesar_per_sender",
    "donor",
    "float",
    "Donor's largest single prior donation as a share of their prior total",
    null_when="donor is unresolved or has no prior donations",
)
def _proporsi_donasi_terbesar(ctx: RuleContext) -> float | None:
    prior = _prior_from_sender(ctx)
    if not prior:
        return None
    total = sum(d.amount_idr for d in prior)
    if total == 0:
        return None
    return max(d.amount_idr for d in prior) / total


@feature(
    "sender_days_since_first_seen",
    "donor",
    "float",
    "Days since the donor's first recorded donation",
    null_when="donor is unresolved or has no prior donations",
)
def _sender_days_since_first_seen(ctx: RuleContext) -> float | None:
    sender = ctx.donation.sender_ref
    if not sender.is_resolved:
        return None
    first = ctx.view.sender_first_seen(sender.key)
    if first is None:
        return None
    return (ctx.donation.occurred_at - first).total_seconds() / 86400


@feature(
    "sender_is_first_donation",
    "donor",
    "bool",
    "This is the donor's first recorded donation",
    null_when="donor is unresolved",
)
def _sender_is_first_donation(ctx: RuleContext) -> bool | None:
    prior = _prior_from_sender(ctx)
    return None if prior is None else len(prior) == 0


@feature(
    "sender_velocity_ratio",
    "donor",
    "float",
    "Donor's 30-day rate against their own 90-day baseline",
    null_when="donor is unresolved or has too little history for a baseline",
)
def _sender_velocity_ratio(ctx: RuleContext) -> float | None:
    sender = ctx.donation.sender_ref
    if not sender.is_resolved:
        return None
    recent = ctx.view.by_sender(
        sender.key,
        since=ctx.donation.occurred_at - timedelta(days=30),
        until=ctx.donation.occurred_at,
        excluding=ctx.donation.donation_id,
    )
    baseline = ctx.view.by_sender(
        sender.key,
        since=ctx.donation.occurred_at - timedelta(days=90),
        until=ctx.donation.occurred_at,
        excluding=ctx.donation.donation_id,
    )
    if len(baseline) < 2:
        return None
    return (len(recent) / 30) / (len(baseline) / 90)


# ---------------------------------------------------------------------------
# Recipient behavioural — strictly prior
# ---------------------------------------------------------------------------


def _prior_to_receiver(ctx: RuleContext, *, days: int | None = None):
    receiver = ctx.donation.receiver_ref
    if not receiver.is_resolved:
        return None
    since = (
        ctx.donation.occurred_at - timedelta(days=days) if days is not None else None
    )
    return ctx.view.by_receiver(
        receiver.key,
        since=since,
        until=ctx.donation.occurred_at,
        excluding=ctx.donation.donation_id,
    )


@feature(
    "sender_unik_per_receiver",
    "recipient",
    "int",
    "Distinct donors the recipient has received from",
    null_when="recipient is unresolved",
)
def _sender_unik_per_receiver(ctx: RuleContext) -> int | None:
    prior = _prior_to_receiver(ctx)
    if prior is None:
        return None
    return len({d.sender_ref.entity_id for d in prior if d.sender_ref.entity_id})


@feature(
    "total_diterima_receiver",
    "recipient",
    "float",
    "Total the recipient has received to date",
    null_when="recipient is unresolved",
)
def _total_diterima_receiver(ctx: RuleContext) -> float | None:
    prior = _prior_to_receiver(ctx)
    return None if prior is None else float(sum(d.amount_idr for d in prior))


@feature(
    "jumlah_transaksi_receiver",
    "recipient",
    "int",
    "Count of donations the recipient has received to date",
    null_when="recipient is unresolved",
)
def _jumlah_transaksi_receiver(ctx: RuleContext) -> int | None:
    prior = _prior_to_receiver(ctx)
    return None if prior is None else len(prior)


@feature(
    "receiver_donor_concentration",
    "recipient",
    "float",
    "Share of the recipient's funding from its largest single donor",
    null_when="recipient is unresolved or has received nothing",
)
def _receiver_donor_concentration(ctx: RuleContext) -> float | None:
    prior = _prior_to_receiver(ctx)
    if not prior:
        return None
    totals: dict[str, int] = {}
    for d in prior:
        key = d.sender_ref.entity_id
        if key:
            totals[key] = totals.get(key, 0) + d.amount_idr
    total = sum(totals.values())
    if total == 0:
        return None
    return max(totals.values()) / total


@feature(
    "receiver_new_donor_ratio_30d",
    "recipient",
    "float",
    "Share of the recipient's last 30 days of donors that are first-time",
    null_when="recipient is unresolved or received nothing in the window",
)
def _receiver_new_donor_ratio_30d(ctx: RuleContext) -> float | None:
    """One of the two features aimed most directly at donation splitting.

    A recipient whose recent donors are overwhelmingly new looks different from
    one with an established base, and neither this nor the amount spread below
    existed in the previous feature set.
    """
    window = _prior_to_receiver(ctx, days=30)
    if not window:
        return None
    since = ctx.donation.occurred_at - timedelta(days=30)
    donors = {d.sender_ref.entity_id for d in window if d.sender_ref.entity_id}
    if not donors:
        return None
    new = sum(1 for s in donors if not ctx.view.has_prior_history(s, before=since))
    return new / len(donors)


@feature(
    "receiver_amount_cv_30d",
    "recipient",
    "float",
    "Coefficient of variation of amounts received in the last 30 days",
    null_when="recipient is unresolved or received fewer than two donations",
)
def _receiver_amount_cv_30d(ctx: RuleContext) -> float | None:
    window = _prior_to_receiver(ctx, days=30)
    if window is None or len(window) < 2:
        return None
    amounts = [d.amount_idr for d in window]
    mean = statistics.fmean(amounts)
    if mean == 0:
        return None
    return statistics.pstdev(amounts) / mean


# ---------------------------------------------------------------------------
# Pair
# ---------------------------------------------------------------------------


def _prior_pair(ctx: RuleContext):
    sender, receiver = ctx.donation.sender_ref, ctx.donation.receiver_ref
    if not (sender.is_resolved and receiver.is_resolved):
        return None
    return ctx.view.by_pair(
        sender.key,
        receiver.key,
        until=ctx.donation.occurred_at,
        excluding=ctx.donation.donation_id,
    )


@feature(
    "pair_prior_count",
    "pair",
    "int",
    "Prior donations between this donor and this recipient",
    null_when="either party is unresolved",
)
def _pair_prior_count(ctx: RuleContext) -> int | None:
    prior = _prior_pair(ctx)
    return None if prior is None else len(prior)


@feature(
    "pair_prior_total",
    "pair",
    "float",
    "Prior cumulative total between this donor and this recipient",
    null_when="either party is unresolved",
)
def _pair_prior_total(ctx: RuleContext) -> float | None:
    """The model-visible companion to the cumulative limit rule.

    The rule produces the legal finding; this lets the classifier learn
    behaviour approaching a limit without being handed the verdict.
    """
    prior = _prior_pair(ctx)
    return None if prior is None else float(sum(d.amount_idr for d in prior))


@feature(
    "pair_is_first",
    "pair",
    "bool",
    "First donation between these two parties",
    null_when="either party is unresolved",
)
def _pair_is_first(ctx: RuleContext) -> bool | None:
    prior = _prior_pair(ctx)
    return None if prior is None else len(prior) == 0


@feature(
    "pair_days_since_last",
    "pair",
    "float",
    "Days since the last donation between these two parties",
    null_when="either party is unresolved, or they have no prior donation",
)
def _pair_days_since_last(ctx: RuleContext) -> float | None:
    prior = _prior_pair(ctx)
    if not prior:
        return None
    last = max(d.occurred_at for d in prior)
    return (ctx.donation.occurred_at - last).total_seconds() / 86400


# ---------------------------------------------------------------------------
# Temporal and contextual
# ---------------------------------------------------------------------------


@feature(
    "is_within_campaign_period",
    "temporal",
    "bool",
    "Donation falls inside a declared campaign period",
    null_when="the electoral context is unknown",
)
def _is_within_campaign_period(ctx: RuleContext) -> bool | None:
    if not ctx.calendar.knows(ctx.donation.electoral_context):
        return None
    return ctx.campaign_period is not None


@feature(
    "campaign_period_phase",
    "temporal",
    "float",
    "Position within the campaign period, 0 at the start and 1 at the end",
    null_when="the donation is not inside a known campaign period",
)
def _campaign_period_phase(ctx: RuleContext) -> float | None:
    period = ctx.campaign_period
    if period is None:
        return None
    span = (period.end - period.start).days
    if span <= 0:
        return None
    return (ctx.donation.occurred_at.date() - period.start).days / span


@feature(
    "days_to_reporting_deadline",
    "temporal",
    "float",
    "Days until the next reporting deadline",
    null_when="no deadline is configured for the electoral context",
)
def _days_to_reporting_deadline(ctx: RuleContext) -> float | None:
    period = ctx.campaign_period
    if period is None or not period.reporting_deadlines:
        return None
    deadline = period.next_deadline_after(ctx.donation.occurred_at.date())
    if deadline is None:
        return None
    return float((deadline - ctx.donation.occurred_at.date()).days)


@feature("day_of_week", "temporal", "int", "Day of week of the donation")
def _day_of_week(ctx: RuleContext) -> int:
    return ctx.donation.occurred_at.weekday()


@feature(
    "hour_of_day",
    "temporal",
    "int",
    "Hour of day of the donation",
    null_when="the source recorded a date without a time",
)
def _hour_of_day(ctx: RuleContext) -> int | None:
    """Null unless the source actually carried a time.

    Scanned forms routinely yield a date alone. Reading midnight off one would
    invent an hour, and enough of them would create a spike at midnight that
    looks like coordinated timing.
    """
    if not ctx.donation.occurred_at_precision.has_time_of_day:
        return None
    return ctx.donation.occurred_at.hour


# ---------------------------------------------------------------------------
# Threshold proximity
#
# Continuous distances to a limit, not the verdict of a limit test. The
# distinction is the whole point: a ratio lets the model learn behaviour near a
# threshold, whereas the statutory outcome would hand it the rule's answer.
# ---------------------------------------------------------------------------


@feature(
    "amount_to_limit_ratio",
    "threshold",
    "float",
    "Amount as a share of the applicable statutory limit",
    null_when="the applicable limit regime is not determinable",
)
def _amount_to_limit_ratio(ctx: RuleContext) -> float | None:
    limit = ctx.applicable_limit
    if limit is None:
        return None
    return ctx.donation.amount_idr / limit.amount_idr


@feature(
    "cumulative_to_limit_ratio",
    "threshold",
    "float",
    "Point-in-time cumulative total as a share of the applicable limit",
    null_when="either party is unresolved or the limit regime is not determinable",
)
def _cumulative_to_limit_ratio(ctx: RuleContext) -> float | None:
    limit = ctx.applicable_limit
    prior = _prior_pair(ctx)
    if limit is None or prior is None:
        return None
    window = ctx.period_window()
    if window is not None:
        prior = tuple(d for d in prior if window.contains(d.occurred_at))
    total = sum(d.amount_idr for d in prior) + ctx.donation.amount_idr
    return total / limit.amount_idr


@feature(
    "in_structuring_band",
    "threshold",
    "bool",
    "Amount sits between 90% and 100% of the applicable limit",
    null_when="the applicable limit regime is not determinable",
)
def _in_structuring_band(ctx: RuleContext) -> bool | None:
    ratio = _amount_to_limit_ratio(ctx)
    return None if ratio is None else 0.90 <= ratio < 1.0


# ---------------------------------------------------------------------------
# Data quality as signal
#
# Quality genuinely correlates with risk, since poor records cluster around
# poor reporting. These also let evaluation notice a model keying on artefacts:
# if extraction confidence dominates importance, it has learned to detect bad
# scans rather than donation splitting.
# ---------------------------------------------------------------------------


@feature(
    "extraction_confidence_min",
    "quality",
    "float",
    "Lowest per-field extraction confidence on this record",
    null_when="no field carries a confidence score",
)
def _extraction_confidence_min(ctx: RuleContext) -> float | None:
    return ctx.donation.extraction_confidence_min


@feature(
    "entity_resolution_confidence",
    "quality",
    "float",
    "Lower of the two parties' resolution confidences",
    null_when="neither party records a resolution confidence",
)
def _entity_resolution_confidence(ctx: RuleContext) -> float | None:
    scores = [
        c
        for c in (
            ctx.donation.sender_ref.resolution_confidence,
            ctx.donation.receiver_ref.resolution_confidence,
        )
        if c is not None
    ]
    return min(scores) if scores else None


@feature(
    "has_unresolved_entity",
    "quality",
    "bool",
    "Either party could not be resolved to a known entity",
)
def _has_unresolved_entity(ctx: RuleContext) -> bool:
    return ctx.donation.has_unresolved_entity


@feature(
    "field_provenance_mix",
    "quality",
    "float",
    "Share of recorded fields that were extracted rather than submitted",
    null_when="the record carries no provenance",
)
def _field_provenance_mix(ctx: RuleContext) -> float | None:
    from cakradana.schema.enums import Provenance

    provenance = ctx.donation.provenance
    if not provenance:
        return None
    extracted = sum(
        1 for p in provenance.values() if p.provenance is Provenance.EXTRACTED
    )
    return extracted / len(provenance)
