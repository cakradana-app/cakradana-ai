"""The graph lane.

Turns structural rule findings into a scored contribution, with reasons an
analyst can check. The network features these findings rest on live in the
features package, so that they load with every other feature rather than only
when this lane happens to be imported.
"""

from __future__ import annotations

from cakradana.lanes.alerts import AlertIndex, AlertKind, GroupAlert
from cakradana.rules.context import RuleContext
from cakradana.rules.engine import RuleEvaluation
from cakradana.scoring.composition import contribution_from
from cakradana.scoring.result import Lane, LaneResult, Reason

#: Behavioural rules whose evidence is structural. Their signals feed this
#: lane; the remaining behavioural rules describe a single party's conduct and
#: inform the classifier instead.
STRUCTURAL_RULES: dict[str, str] = {
    "RULE-T2-01": "FAN_IN_BURST",
    "RULE-T2-03": "FAN_OUT",
    "RULE-T2-04": "PASS_THROUGH",
    "RULE-T2-09": "DONOR_CONCENTRATION",
}

# ---------------------------------------------------------------------------
# Lane
# ---------------------------------------------------------------------------

#: Typical distinct-sender count for a recipient over a fortnight. A reason
#: without a reference point is not actionable, and this is the reference the
#: fan-in statement compares against until measured values replace it.
TYPICAL_FAN_IN = 3


class GraphLane:
    """Turns structural rule signals and group alerts into a contribution.

    Two sources feed it. The per-donation structural rules ask whether *this*
    donation sits in a suspicious shape; the group alerts describe the shape
    itself. A donation inside a detected cluster inherits the cluster's score,
    because the evidence against it is the cluster — reasoning about it alone
    is exactly the mistake group alerts exist to prevent.
    """

    name = Lane.GRAPH

    def __init__(self, alerts: AlertIndex | None = None) -> None:
        self.alerts = alerts or AlertIndex()

    def use(self, alerts: AlertIndex) -> None:
        """Adopt a freshly detected set of clusters.

        Detection runs over the whole population, not per donation, so a
        cluster becomes visible only once enough of it has arrived. A donation
        scored before its cluster existed keeps the score it was given; the
        record of what changed is the rescoring event, not an edit to the old
        one.
        """
        self.alerts = alerts

    def evaluate(self, evaluation: RuleEvaluation, ctx: RuleContext) -> LaneResult:
        fired = [
            signal
            for signal in evaluation.behavioural_signals
            if signal.rule_id in STRUCTURAL_RULES
        ]
        covering = self.alerts.covering(ctx.donation.donation_id)
        if not fired and not covering:
            return contribution_from(Lane.GRAPH, 0.0, ())

        reasons = tuple(self._reason(signal, ctx) for signal in fired) + tuple(
            self._alert_reason(alert) for alert in covering
        )

        # Intensity rises with the number of independent structural findings
        # but saturates: three patterns firing together is materially more than
        # one, and six is not materially more than three.
        from_rules = 0.0
        if fired:
            weighted = max((signal.label_weight or 0.5) for signal in fired)
            from_rules = min(len(fired) / 3.0, 1.0) * weighted / 0.6

        # Taken as the stronger of the two rather than their sum. A cluster
        # detected by the group pass and by the per-donation rule is one
        # observation seen twice, and adding it to itself would rank a
        # doubly-detected pattern above a worse one detected once.
        from_alerts = max((alert.score / 100 for alert in covering), default=0.0)
        return contribution_from(Lane.GRAPH, max(from_rules, from_alerts), reasons)

    def _alert_reason(self, alert: GroupAlert) -> Reason:
        """States what the cluster is, and that the donation is part of it.

        Phrased as a fact about a set of payments. The alert names no motive
        and asserts nothing about the parties, because the same shape is
        produced by a coordinated split and by a successful fundraising drive.
        """
        counterparties = len(alert.subject.counterparties)
        donations = len(alert.subject.donations)
        window = alert.subject.window
        span = (window.to - window.from_).days
        if alert.kind is AlertKind.FAN_OUT:
            statement = (
                f"This donation is one of {donations} from the same donor to "
                f"{counterparties} distinct recipients within {span} days."
            )
        elif alert.kind is AlertKind.LAYERING_CHAIN:
            # Three conclusions were packed into the previous sentence:
            # "layering" is a term of art for a deliberate laundering stage,
            # "chain" asserts the donations are connected rather than merely
            # co-occurring, and "intermediate entities" assigns a role to
            # parties who may simply have donated. An analyst reading the
            # sequence may well conclude layering; the reason must not conclude
            # it for them, and a party described as intermediate in a case note
            # has been characterised by the system rather than by a person.
            statement = (
                f"This donation is one of {donations} that occurred in "
                f"sequence between {counterparties} other parties within "
                f"{span} days."
            )
        else:
            statement = (
                f"This donation is one of {donations} reaching the same "
                f"recipient from {counterparties} distinct donors within "
                f"{span} days."
            )
        if alert.provisional_node_ratio > 0:
            statement += (
                f" {alert.provisional_node_ratio:.0%} of the group rests on "
                f"parties that could not be resolved to a known entity."
            )
        return Reason(
            code=str(alert.kind),
            lane=Lane.GRAPH,
            weight=alert.score / 100,
            statement=statement,
            comparison=alert.comparison,
            # Points at the group, not at this donation. An analyst following
            # it should arrive at the pattern rather than back at the single
            # payment that cannot justify anything on its own.
            evidence_ref=alert.alert_id,
        )

    def _reason(self, signal, ctx: RuleContext) -> Reason:
        code = STRUCTURAL_RULES[signal.rule_id]
        return Reason(
            code=code,
            lane=Lane.GRAPH,
            weight=min(signal.label_weight or 0.5, 1.0),
            statement=signal.explanation or signal.rule_id,
            comparison=self._comparison(code, signal),
            evidence_ref=f"donation:{ctx.donation.donation_id}",
        )

    @staticmethod
    def _comparison(code: str, signal) -> str | None:
        if code == "FAN_IN_BURST":
            return (
                f"A recipient more usually draws around {TYPICAL_FAN_IN} distinct "
                f"donors in a window of this length."
            )
        if code == "PASS_THROUGH":
            return (
                "Most donors have no comparable inflow shortly before they give."
            )
        if code == "DONOR_CONCENTRATION":
            return "Most recipients draw the bulk of their funding from many donors."
        if code == "FAN_OUT":
            return "Most donors give to one or two recipients."
        return None
