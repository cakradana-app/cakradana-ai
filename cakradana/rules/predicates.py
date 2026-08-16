"""Rule tests.

Each test kind is a registered, typed implementation. Rules stay data — a
threshold or a window is edited in YAML — but the space of tests stays
reviewable, because adding one is a code change with tests attached. An
expression language would let an unreviewed string decide what counts as a
statutory violation.

Tier-1 and Tier-2 tests fail differently, deliberately.

A statutory test that cannot evaluate one of its conditions returns
indeterminate. It is asserting a fact about the law, and a fact asserted on
incomplete information is not a weaker fact, it is a wrong one.

A behavioural test evaluates the conditions it can and records which it had to
skip. These are hypotheses whose purpose is to generate training labels and
rank suspicion; refusing to produce any signal because one sub-condition was
unavailable would leave the classifier with nothing to learn from, and the
skipped conditions travel with the result so nothing is overstated.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable

from cakradana.registers import RegisterSet
from cakradana.rules.context import RuleContext
from cakradana.rules.schema import Rule
from cakradana.schema.enums import EntityType

#: Names commonly standing in for an absent donor identity on Indonesian
#: donation records. Matching one is evidence that identity was not recorded,
#: which is a statutory concern in itself.
PLACEHOLDER_DONOR_NAMES = frozenset(
    {
        "nn",
        "n.n.",
        "anonim",
        "anonymous",
        "hamba allah",
        "hamba tuhan",
        "tidak diketahui",
        "tanpa nama",
        "-",
        "?",
    }
)


@dataclass
class PredicateResult:
    """Outcome of one test."""

    fired: bool = False
    indeterminate: str | None = None
    observed: float | int | None = None
    threshold: float | int | None = None
    #: Structured detail an analyst or a reason template can use.
    facts: dict[str, object] = field(default_factory=dict)
    #: Conditions a behavioural test could not evaluate. Carried so a signal
    #: is never read as stronger than the evidence behind it.
    skipped: tuple[str, ...] = ()

    @classmethod
    def undetermined(cls, reason: str) -> PredicateResult:
        return cls(fired=False, indeterminate=reason)


Predicate = Callable[[Rule, RuleContext], PredicateResult]

_REGISTRY: dict[str, Predicate] = {}


def predicate(kind: str) -> Callable[[Predicate], Predicate]:
    def register(fn: Predicate) -> Predicate:
        if kind in _REGISTRY:
            raise ValueError(f"duplicate predicate kind {kind!r}")
        _REGISTRY[kind] = fn
        return fn

    return register


def get_predicate(kind: str) -> Predicate:
    try:
        return _REGISTRY[kind]
    except KeyError:
        raise KeyError(
            f"unknown rule test kind {kind!r}; rules may only use registered tests"
        ) from None


def known_kinds() -> frozenset[str]:
    return frozenset(_REGISTRY)


# ---------------------------------------------------------------------------
# Tier 1 — statutory
# ---------------------------------------------------------------------------


@predicate("threshold")
def _threshold(rule: Rule, ctx: RuleContext) -> PredicateResult:
    """A single donation exceeds the limit applicable to it."""
    limit = ctx.applicable_limit
    if limit is None:
        return PredicateResult.undetermined(
            "the applicable limit regime could not be determined for this donation"
        )
    amount = ctx.donation.amount_idr
    return PredicateResult(
        fired=amount > limit.amount_idr,
        observed=amount,
        threshold=limit.amount_idr,
        facts={
            "amount": amount,
            "threshold": limit.amount_idr,
            "regime": str(limit.regime),
        },
    )


@predicate("aggregate_threshold")
def _aggregate_threshold(rule: Rule, ctx: RuleContext) -> PredicateResult:
    """Donations accumulating past the limit over a statutory period.

    This is the test the previous implementation lacked entirely: it compared a
    single row's amount against the cap, so twenty donations at the cap each
    passed while together breaching it twentyfold.

    The sum includes the donation being scored, because the finding attaches to
    a donation that is itself part of the excess. Donations after the crossing
    are reported too — each is also above the cap — but are marked as
    continuations so triage can tell them from the crossing itself.
    """
    params = rule.test.params()
    group_by = tuple(params.get("group_by", ("resolved_sender_id", "resolved_receiver_id")))

    donation = ctx.donation
    if not donation.sender_ref.is_resolved:
        return PredicateResult.undetermined(
            "donor identity is unresolved, so donations cannot be accumulated "
            "against one donor"
        )
    if "resolved_receiver_id" in group_by and not donation.receiver_ref.is_resolved:
        return PredicateResult.undetermined(
            "recipient identity is unresolved, so donations cannot be "
            "accumulated against one recipient"
        )

    limit = ctx.applicable_limit
    if limit is None:
        return PredicateResult.undetermined(
            "the applicable limit regime could not be determined for this donation"
        )
    window = ctx.period_window()
    if window is None:
        return PredicateResult.undetermined(
            "the statutory period covering this donation is not known"
        )

    sender_id = donation.sender_ref.key
    if "resolved_receiver_id" in group_by:
        prior = ctx.view.by_pair(
            sender_id,
            donation.receiver_ref.key,
            since=window.start,
            until=window.end,
            excluding=donation.donation_id,
        )
        scope = "to this recipient"
    else:
        prior = ctx.view.by_sender(
            sender_id,
            since=window.start,
            until=window.end,
            excluding=donation.donation_id,
        )
        scope = "across all recipients"

    prior_total = sum(d.amount_idr for d in prior)
    total = prior_total + donation.amount_idr

    return PredicateResult(
        fired=total > limit.amount_idr,
        observed=total,
        threshold=limit.amount_idr,
        facts={
            "total": total,
            "prior_total": prior_total,
            "threshold": limit.amount_idr,
            "contributing_donations": len(prior) + 1,
            "period_label": window.label,
            "scope": scope,
            "is_crossing": prior_total <= limit.amount_idr < total,
            "regime": str(limit.regime),
        },
    )


@predicate("identity_absent")
def _identity_absent(rule: Rule, ctx: RuleContext) -> PredicateResult:
    """Donor identity is missing, placeholder, or unusable.

    Extraction must record *why* identity is absent. A form that was genuinely
    submitted anonymously and a scan the reader could not decipher produce the
    same empty field, and they are legally different: the first is an offence
    by the recipient, the second is a data-quality problem. Reporting them
    alike would manufacture accusations out of poor scans.
    """
    sender = ctx.donation.sender_ref
    reasons: list[str] = []

    raw = (sender.raw_text or "").strip()
    if not sender.is_resolved and not raw:
        reasons.append("no donor identity recorded")
    elif raw.casefold() in PLACEHOLDER_DONOR_NAMES:
        reasons.append(f"donor recorded as a placeholder ({raw!r})")

    if sender.entity_type is EntityType.UNKNOWN and not sender.is_resolved:
        reasons.append("donor type could not be established")

    if not reasons:
        return PredicateResult(fired=False)

    provenance = ctx.donation.provenance.get("sender_ref")
    quality = provenance.reason if provenance else None
    if quality is None:
        return PredicateResult.undetermined(
            "donor identity is absent, but the record does not say whether it "
            "was withheld at source or lost in extraction; these are different "
            "findings and must not be merged"
        )

    return PredicateResult(
        fired=quality == "declared_anonymous",
        facts={"reasons": tuple(reasons), "source_quality": quality},
    )


@predicate("register_membership")
def _register_membership(rule: Rule, ctx: RuleContext) -> PredicateResult:
    """The donor appears on a register of prohibited sources.

    A lookup, never an inference. Name patterns may nominate candidates for the
    register but may not produce a finding.
    """
    params = rule.test.params()
    register_name = str(params.get("register", RegisterSet.PROHIBITED_SOURCE))
    sender = ctx.donation.sender_ref

    result = ctx.registers.lookup(
        register_name,
        sender.entity_id,
        when=ctx.donation.occurred_at.date(),
        now=ctx.now,
        entity_name=sender.raw_text,
    )
    if not result.available:
        return PredicateResult.undetermined(
            result.reason or f"register {register_name!r} is unavailable"
        )
    return PredicateResult(
        fired=result.member,
        facts={
            "register": register_name,
            "category": result.entry.category if result.entry else None,
            "matched_name": result.entry.canonical_name if result.entry else None,
        },
    )


@predicate("foreign_jurisdiction")
def _foreign_jurisdiction(rule: Rule, ctx: RuleContext) -> PredicateResult:
    """The donor is foreign.

    Jurisdiction is carried on the resolved entity or it is not known. It is
    never inferred from a name: name-based nationality inference is unreliable
    and discriminatory, and here it would attach a statutory offence to someone
    on the basis of what they are called.
    """
    params = rule.test.params()
    domestic = str(params.get("domestic_code", "ID"))
    sender = ctx.sender_entity()
    jurisdiction = sender.jurisdiction if sender else None

    if jurisdiction is None:
        return PredicateResult.undetermined(
            "donor jurisdiction is not recorded, and nationality must not be "
            "inferred from a name"
        )
    return PredicateResult(
        fired=jurisdiction != domestic,
        facts={"jurisdiction": jurisdiction, "domestic_code": domestic},
    )


@predicate("report_reconciliation")
def _report_reconciliation(rule: Rule, ctx: RuleContext) -> PredicateResult:
    """The donation is missing from the recipient's filed report.

    Detects an offence currently invisible to every party, and stays
    indeterminate until campaign finance submissions and designated-account
    transactions are actually available to compare against.
    """
    return PredicateResult.undetermined(
        "campaign finance report and designated-account data are not available "
        "for reconciliation"
    )


# ---------------------------------------------------------------------------
# Tier 2 — behavioural
# ---------------------------------------------------------------------------


@predicate("fan_in_burst")
def _fan_in_burst(rule: Rule, ctx: RuleContext) -> PredicateResult:
    """Many distinct donors converging on one recipient in a short window.

    The structural signature of splitting one source into many nominal donors.
    The thin-donor share matters as much as the count: genuine grassroots
    fundraising also produces fan-in, and without that condition this test
    would flag every successful public appeal.
    """
    params = rule.test.params()
    window_days = int(params.get("window_days", 14))
    min_senders = int(params.get("min_distinct_senders", 15))
    max_amount_ratio = params.get("max_amount_ratio", 0.25)
    min_thin_ratio = float(params.get("min_thin_sender_ratio", 0.5))

    donation = ctx.donation
    if not donation.receiver_ref.is_resolved:
        return PredicateResult.undetermined("recipient identity is unresolved")

    since = donation.occurred_at - timedelta(days=window_days)
    window = ctx.view.by_receiver(
        donation.receiver_ref.key, since=since, until=donation.occurred_at
    )
    senders = {
        d.sender_ref.entity_id for d in window if d.sender_ref.entity_id is not None
    }
    senders.add(donation.sender_ref.entity_id or "")
    senders.discard("")
    distinct = len(senders)

    thin = sum(
        1
        for s in senders
        if not ctx.view.has_prior_history(s, before=since)
    )
    thin_ratio = thin / distinct if distinct else 0.0

    skipped: list[str] = []
    amounts_below = True
    limit = ctx.applicable_limit
    if max_amount_ratio is not None:
        if limit is None:
            skipped.append("amount ceiling: applicable limit not determinable")
        else:
            ceiling = float(max_amount_ratio) * limit.amount_idr
            amounts_below = all(d.amount_idr < ceiling for d in window) and (
                donation.amount_idr < ceiling
            )

    fired = (
        distinct >= min_senders and thin_ratio >= min_thin_ratio and amounts_below
    )
    return PredicateResult(
        fired=fired,
        observed=distinct,
        threshold=min_senders,
        skipped=tuple(skipped),
        facts={
            "distinct_senders": distinct,
            "window_days": window_days,
            "thin_senders": thin,
            "thin_ratio": round(thin_ratio, 3),
        },
    )


@predicate("amount_band")
def _amount_band(rule: Rule, ctx: RuleContext) -> PredicateResult:
    """The amount sits just under the applicable limit.

    Deliberate positioning below a threshold is the defining shape of
    structuring. The limit is the entire reference point, so unlike the other
    behavioural tests this one cannot degrade when the regime is unknown.
    """
    params = rule.test.params()
    low = float(params.get("low_ratio", 0.90))
    high = float(params.get("high_ratio", 0.999))

    limit = ctx.applicable_limit
    if limit is None:
        return PredicateResult.undetermined(
            "the applicable limit regime could not be determined, so proximity "
            "to a limit is not defined"
        )
    ratio = ctx.donation.amount_idr / limit.amount_idr
    return PredicateResult(
        fired=low <= ratio <= high,
        observed=round(ratio, 4),
        facts={
            "ratio_to_limit": round(ratio, 4),
            "band": [low, high],
            "threshold": limit.amount_idr,
        },
    )


@predicate("fan_out")
def _fan_out(rule: Rule, ctx: RuleContext) -> PredicateResult:
    """One donor spreading across many recipients in a short window."""
    params = rule.test.params()
    window_days = int(params.get("window_days", 14))
    min_receivers = int(params.get("min_distinct_receivers", 10))

    donation = ctx.donation
    if not donation.sender_ref.is_resolved:
        return PredicateResult.undetermined("donor identity is unresolved")

    since = donation.occurred_at - timedelta(days=window_days)
    receivers = ctx.view.distinct_receivers_from(donation.sender_ref.key, since=since)
    receivers.add(donation.receiver_ref.entity_id or "")
    receivers.discard("")
    distinct = len(receivers)

    return PredicateResult(
        fired=distinct >= min_receivers,
        observed=distinct,
        threshold=min_receivers,
        facts={"distinct_receivers": distinct, "window_days": window_days},
    )


@predicate("pass_through")
def _pass_through(rule: Rule, ctx: RuleContext) -> PredicateResult:
    """Money arriving and leaving an intermediary at similar size and time.

    An entity that receives and forwards comparable amounts within days, with
    no other economic footprint, is behaving like a conduit rather than a donor.
    """
    params = rule.test.params()
    window_days = int(params.get("window_days", 7))
    low = float(params.get("low_ratio", 0.85))
    high = float(params.get("high_ratio", 1.15))

    donation = ctx.donation
    if not donation.sender_ref.is_resolved:
        return PredicateResult.undetermined("donor identity is unresolved")

    sender_id = donation.sender_ref.key
    since = donation.occurred_at - timedelta(days=window_days)
    inflows = ctx.view.by_receiver(
        sender_id, since=since, until=donation.occurred_at
    )
    if not inflows:
        return PredicateResult(
            fired=False, facts={"inflow_total": 0, "window_days": window_days}
        )

    inflow_total = sum(d.amount_idr for d in inflows)
    ratio = donation.amount_idr / inflow_total if inflow_total else 0.0
    lag_days = min(
        (donation.occurred_at - d.occurred_at).days for d in inflows
    )

    return PredicateResult(
        fired=low <= ratio <= high,
        observed=round(ratio, 4),
        facts={
            "inflow_total": inflow_total,
            "outflow": donation.amount_idr,
            "ratio": round(ratio, 4),
            "lag_days": lag_days,
            "window_days": window_days,
        },
    )


@predicate("self_funded_inflow")
def _self_funded_inflow(rule: Rule, ctx: RuleContext) -> PredicateResult:
    """Declared self-funding shortly preceded by an unexplained inflow.

    Stays indeterminate until submissions distinguish a candidate's own
    contribution from a third party's, because without that distinction there
    is nothing to test against.
    """
    params = rule.test.params()
    window_days = int(params.get("window_days", 30))
    min_ratio = float(params.get("min_inflow_ratio", 0.5))

    donation = ctx.donation
    if donation.is_self_funded_declared is None:
        return PredicateResult.undetermined(
            "records do not distinguish a candidate's own funds from "
            "third-party donations"
        )
    if not donation.is_self_funded_declared:
        return PredicateResult(fired=False)
    if not donation.sender_ref.is_resolved:
        return PredicateResult.undetermined("donor identity is unresolved")

    since = donation.occurred_at - timedelta(days=window_days)
    inflows = ctx.view.by_receiver(
        donation.sender_ref.key, since=since, until=donation.occurred_at
    )
    inflow_total = sum(d.amount_idr for d in inflows)
    ratio = inflow_total / donation.amount_idr if donation.amount_idr else 0.0

    return PredicateResult(
        fired=ratio >= min_ratio,
        observed=round(ratio, 4),
        facts={"inflow_total": inflow_total, "ratio": round(ratio, 4)},
    )


@predicate("velocity_spike")
def _velocity_spike(rule: Rule, ctx: RuleContext) -> PredicateResult:
    """A donor giving far faster than their own established rate."""
    params = rule.test.params()
    recent_days = int(params.get("recent_days", 30))
    baseline_days = int(params.get("baseline_days", 90))
    multiple = float(params.get("min_multiple", 5.0))

    donation = ctx.donation
    if not donation.sender_ref.is_resolved:
        return PredicateResult.undetermined("donor identity is unresolved")

    sender_id = donation.sender_ref.key
    recent = ctx.view.by_sender(
        sender_id,
        since=donation.occurred_at - timedelta(days=recent_days),
        until=donation.occurred_at,
    )
    baseline = ctx.view.by_sender(
        sender_id,
        since=donation.occurred_at - timedelta(days=baseline_days),
        until=donation.occurred_at,
    )
    if len(baseline) < 2:
        return PredicateResult(
            fired=False,
            skipped=("baseline: donor has too little history to have a rate",),
            facts={"baseline_donations": len(baseline)},
        )

    recent_rate = len(recent) / recent_days
    baseline_rate = len(baseline) / baseline_days
    ratio = recent_rate / baseline_rate if baseline_rate else 0.0

    return PredicateResult(
        fired=ratio >= multiple,
        observed=round(ratio, 3),
        threshold=multiple,
        facts={"recent_donations": len(recent), "baseline_donations": len(baseline)},
    )


@predicate("amount_homogeneity")
def _amount_homogeneity(rule: Rule, ctx: RuleContext) -> PredicateResult:
    """Donations to one recipient clustered tightly around one value.

    Independent donors choose varied amounts. A tight cluster suggests one
    decision executed many times.
    """
    params = rule.test.params()
    window_days = int(params.get("window_days", 30))
    min_count = int(params.get("min_donations", 10))
    max_cv = float(params.get("max_coefficient_of_variation", 0.1))

    donation = ctx.donation
    if not donation.receiver_ref.is_resolved:
        return PredicateResult.undetermined("recipient identity is unresolved")

    since = donation.occurred_at - timedelta(days=window_days)
    window = ctx.view.by_receiver(
        donation.receiver_ref.key, since=since, until=donation.occurred_at
    )
    amounts = [d.amount_idr for d in window] + [donation.amount_idr]
    if len(amounts) < min_count:
        return PredicateResult(fired=False, facts={"donations": len(amounts)})

    mean = statistics.fmean(amounts)
    if mean == 0:
        return PredicateResult(fired=False)
    cv = statistics.pstdev(amounts) / mean

    return PredicateResult(
        fired=cv < max_cv,
        observed=round(cv, 4),
        threshold=max_cv,
        facts={"donations": len(amounts), "coefficient_of_variation": round(cv, 4)},
    )


@predicate("thin_donor")
def _thin_donor(rule: Rule, ctx: RuleContext) -> PredicateResult:
    """A donor's first appearance is a large donation."""
    params = rule.test.params()
    min_ratio = float(params.get("min_limit_ratio", 0.5))

    donation = ctx.donation
    if not donation.sender_ref.is_resolved:
        return PredicateResult.undetermined("donor identity is unresolved")

    prior = ctx.view.by_sender(
        donation.sender_ref.key,
        until=donation.occurred_at,
        excluding=donation.donation_id,
    )
    if prior:
        return PredicateResult(fired=False, facts={"prior_donations": len(prior)})

    limit = ctx.applicable_limit
    if limit is None:
        return PredicateResult(
            fired=False,
            skipped=("size: applicable limit not determinable",),
            facts={"prior_donations": 0},
        )
    ratio = donation.amount_idr / limit.amount_idr
    return PredicateResult(
        fired=ratio > min_ratio,
        observed=round(ratio, 4),
        threshold=min_ratio,
        facts={"prior_donations": 0, "ratio_to_limit": round(ratio, 4)},
    )


@predicate("donor_concentration")
def _donor_concentration(rule: Rule, ctx: RuleContext) -> PredicateResult:
    """One donor supplying most of a recipient's funding."""
    params = rule.test.params()
    min_share = float(params.get("min_share", 0.6))
    min_total = int(params.get("min_recipient_total_idr", 0))

    donation = ctx.donation
    if not (donation.receiver_ref.is_resolved and donation.sender_ref.is_resolved):
        return PredicateResult.undetermined("donor or recipient identity is unresolved")

    window = ctx.period_window()
    since = window.start if window else None
    until = window.end if window else donation.occurred_at

    received = ctx.view.by_receiver(
        donation.receiver_ref.key,
        since=since,
        until=until,
        excluding=donation.donation_id,
    )
    total = sum(d.amount_idr for d in received) + donation.amount_idr
    from_this_donor = (
        sum(
            d.amount_idr
            for d in received
            if d.sender_ref.entity_id == donation.sender_ref.key
        )
        + donation.amount_idr
    )
    if total < min_total or total == 0:
        return PredicateResult(fired=False, facts={"recipient_total": total})

    share = from_this_donor / total
    return PredicateResult(
        fired=share > min_share,
        observed=round(share, 4),
        threshold=min_share,
        facts={
            "recipient_total": total,
            "from_this_donor": from_this_donor,
            "share": round(share, 4),
        },
    )


@predicate("deadline_clustering")
def _deadline_clustering(rule: Rule, ctx: RuleContext) -> PredicateResult:
    """Donations bunched immediately before a reporting deadline."""
    params = rule.test.params()
    hours = int(params.get("window_hours", 72))
    multiple = float(params.get("min_multiple", 3.0))
    baseline_days = int(params.get("baseline_days", 30))

    donation = ctx.donation
    if not donation.receiver_ref.is_resolved:
        return PredicateResult.undetermined("recipient identity is unresolved")

    period = ctx.campaign_period
    if period is None or not period.reporting_deadlines:
        return PredicateResult.undetermined(
            "no reporting deadline is configured for this electoral context"
        )
    deadline = period.next_deadline_after(donation.occurred_at.date())
    if deadline is None:
        return PredicateResult(fired=False, facts={"deadline": None})

    deadline_at = datetime.combine(
        deadline, datetime.max.time(), tzinfo=donation.occurred_at.tzinfo
    )
    window_start = deadline_at - timedelta(hours=hours)
    if not (window_start <= donation.occurred_at <= deadline_at):
        return PredicateResult(fired=False, facts={"deadline": str(deadline)})

    in_window = ctx.view.by_receiver(
        donation.receiver_ref.key, since=window_start, until=donation.occurred_at
    )
    baseline = ctx.view.by_receiver(
        donation.receiver_ref.key,
        since=deadline_at - timedelta(days=baseline_days),
        until=window_start,
    )
    window_rate = (len(in_window) + 1) / (hours / 24)
    baseline_rate = len(baseline) / max(baseline_days - hours / 24, 1)
    if baseline_rate == 0:
        return PredicateResult(
            fired=False,
            skipped=("baseline: recipient has no donations before the window",),
            facts={"deadline": str(deadline), "in_window": len(in_window) + 1},
        )

    ratio = window_rate / baseline_rate
    return PredicateResult(
        fired=ratio >= multiple,
        observed=round(ratio, 3),
        threshold=multiple,
        facts={
            "deadline": str(deadline),
            "in_window": len(in_window) + 1,
            "rate_multiple": round(ratio, 3),
        },
    )
