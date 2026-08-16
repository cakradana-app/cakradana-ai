"""Network position features.

The previous feature set claimed to carry centrality and did not: two of its
columns were labelled as centrality measures while being verbatim copies of two
counting columns already present. The effective feature count was twelve rather
than sixteen, and nothing in it described network position at all.

These compute degrees, pass-through behaviour, component size, and shared
counterparties for real, each bound to what was knowable at the donation's own
date. Structure is where this domain's hardest patterns live: splitting a
contribution shows up as convergence and routing one through an intermediary
shows up as a matched inflow and outflow, and neither is visible in any single
donation's own fields.
"""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import timedelta

from cakradana.features.definitions import feature
from cakradana.rules.context import RuleContext


# ---------------------------------------------------------------------------
# Network features
# ---------------------------------------------------------------------------


@feature(
    "sender_out_degree",
    "graph",
    "int",
    "Distinct recipients this donor has reached",
    null_when="donor is unresolved",
)
def _sender_out_degree(ctx: RuleContext) -> int | None:
    sender = ctx.donation.sender_ref
    if not sender.is_resolved:
        return None
    return len(ctx.view.distinct_receivers_from(sender.key))


@feature(
    "receiver_in_degree",
    "graph",
    "int",
    "Distinct donors this recipient has drawn",
    null_when="recipient is unresolved",
)
def _receiver_in_degree(ctx: RuleContext) -> int | None:
    receiver = ctx.donation.receiver_ref
    if not receiver.is_resolved:
        return None
    return len(ctx.view.distinct_senders_to(receiver.key))


@feature(
    "receiver_in_degree_velocity_14d",
    "graph",
    "float",
    "New distinct donors in 14 days as a share of the recipient's total",
    null_when="recipient is unresolved or has no donors yet",
)
def _receiver_in_degree_velocity(ctx: RuleContext) -> float | None:
    """How fast a recipient's donor base is widening.

    A recipient acquiring most of its donors inside a fortnight looks
    different from one that accumulated them over a year, and the shape of
    that curve is what separates a burst from a base.
    """
    receiver = ctx.donation.receiver_ref
    if not receiver.is_resolved:
        return None
    total = ctx.view.distinct_senders_to(receiver.key)
    if not total:
        return None
    recent = ctx.view.distinct_senders_to(
        receiver.key, since=ctx.donation.occurred_at - timedelta(days=14)
    )
    return len(recent) / len(total)


@feature(
    "pass_through_ratio",
    "graph",
    "float",
    "This donation as a share of what the donor received in the prior week",
    null_when="donor is unresolved or received nothing in the window",
)
def _pass_through_ratio(ctx: RuleContext) -> float | None:
    """Whether the donor is forwarding rather than giving.

    An entity that receives an amount and passes on something close to it,
    days later, is behaving like a conduit. The ratio is the signal; the
    identity of what it conceals is not something this system claims to know.
    """
    sender = ctx.donation.sender_ref
    if not sender.is_resolved:
        return None
    inflows = ctx.view.by_receiver(
        sender.key,
        since=ctx.donation.occurred_at - timedelta(days=7),
        until=ctx.donation.occurred_at,
    )
    total = sum(d.amount_idr for d in inflows)
    if total == 0:
        return None
    return ctx.donation.amount_idr / total


@feature(
    "pass_through_lag_days",
    "graph",
    "float",
    "Days between the donor's most recent inflow and this donation",
    null_when="donor is unresolved or received nothing in the window",
)
def _pass_through_lag_days(ctx: RuleContext) -> float | None:
    sender = ctx.donation.sender_ref
    if not sender.is_resolved:
        return None
    inflows = ctx.view.by_receiver(
        sender.key,
        since=ctx.donation.occurred_at - timedelta(days=7),
        until=ctx.donation.occurred_at,
    )
    if not inflows:
        return None
    latest = max(d.occurred_at for d in inflows)
    return (ctx.donation.occurred_at - latest).total_seconds() / 86400


@feature(
    "shared_counterparty_count",
    "graph",
    "int",
    "Other donors that have also given to this recipient",
    null_when="either party is unresolved",
)
def _shared_counterparty_count(ctx: RuleContext) -> int | None:
    sender, receiver = ctx.donation.sender_ref, ctx.donation.receiver_ref
    if not (sender.is_resolved and receiver.is_resolved):
        return None
    others = ctx.view.distinct_senders_to(receiver.key)
    others.discard(sender.key)
    return len(others)


@feature(
    "component_size",
    "graph",
    "int",
    "Size of the connected component the two parties sit in",
    null_when="either party is unresolved",
)
def _component_size(ctx: RuleContext) -> int | None:
    """How large a cluster this donation belongs to.

    Bounded rather than exhaustive: the search stops once the component is
    clearly large, because the distinction that matters is between a small
    isolated cluster and a well-connected one, not between two large numbers.
    """
    sender, receiver = ctx.donation.sender_ref, ctx.donation.receiver_ref
    if not (sender.is_resolved and receiver.is_resolved):
        return None
    return _bounded_component_size(ctx, {sender.key, receiver.key}, limit=500)


def _bounded_component_size(
    ctx: RuleContext, seeds: set[str], *, limit: int
) -> int:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for donation in ctx.view:
        a = donation.sender_ref.entity_id
        b = donation.receiver_ref.entity_id
        if a and b:
            adjacency[a].add(b)
            adjacency[b].add(a)

    seen: set[str] = set()
    queue = deque(seeds)
    while queue and len(seen) < limit:
        node = queue.popleft()
        if node in seen:
            continue
        seen.add(node)
        queue.extend(adjacency[node] - seen)
    return len(seen)


