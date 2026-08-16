"""The graph lane.

Turns structural rule findings into a scored contribution, with reasons an
analyst can check. The network features these findings rest on live in the
features package, so that they load with every other feature rather than only
when this lane happens to be imported.
"""

from __future__ import annotations

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
    """Turns structural rule signals into a scored contribution."""

    name = Lane.GRAPH

    def evaluate(self, evaluation: RuleEvaluation, ctx: RuleContext) -> LaneResult:
        fired = [
            signal
            for signal in evaluation.behavioural_signals
            if signal.rule_id in STRUCTURAL_RULES
        ]
        if not fired:
            return contribution_from(Lane.GRAPH, 0.0, ())

        reasons = tuple(self._reason(signal, ctx) for signal in fired)
        # Intensity rises with the number of independent structural findings
        # but saturates: three patterns firing together is materially more than
        # one, and six is not materially more than three.
        intensity = min(len(fired) / 3.0, 1.0)
        weighted = max(
            (signal.label_weight or 0.5) for signal in fired
        )
        return contribution_from(
            Lane.GRAPH, intensity * weighted / 0.6, reasons
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
