"""Scoring a donation end to end.

Order matters here. Statutory rules run first and their findings stand on their
own, independent of anything the lanes conclude. Features are computed next,
and the behavioural lanes run over them.

That order is also the failure order. Legal findings are computable from the
donation and the rule set alone, so the system's most defensible output
survives the loss of the feature store, the model, and every lane. A lane that
cannot run reports that it could not, and the score says it is incomplete
rather than quietly redistributing the missing points.
"""

from __future__ import annotations

from datetime import datetime
from typing import Mapping, Protocol, Sequence

from cakradana.calendar import ElectoralCalendar
from cakradana.features import FeatureService, FeatureVector
from cakradana.history import PointInTimeView
from cakradana.lanes.alerts import AlertIndex
from cakradana.lanes.graph import GraphLane
from cakradana.registers import RegisterSet
from cakradana.rules import RuleEngine, RuleSet
from cakradana.rules.context import RuleContext
from cakradana.rules.engine import RuleEvaluation
from cakradana.scoring.composition import ScoreComposer, unavailable
from cakradana.scoring.result import Lane, LaneResult, ScoringResult
from cakradana.schema import Donation, Entity


class DetectionLane(Protocol):
    """A source of behavioural suspicion."""

    name: Lane

    def evaluate(
        self,
        evaluation: RuleEvaluation,
        ctx: RuleContext,
        features: FeatureVector,
    ) -> LaneResult: ...


class GraphLaneAdapter:
    """Adapts the graph lane, which needs no feature vector, to the protocol."""

    name = Lane.GRAPH

    def __init__(self, alerts: AlertIndex | None = None) -> None:
        self._lane = GraphLane(alerts)

    def use(self, alerts: AlertIndex) -> None:
        self._lane.use(alerts)

    @property
    def alerts(self) -> AlertIndex:
        return self._lane.alerts

    def evaluate(
        self,
        evaluation: RuleEvaluation,
        ctx: RuleContext,
        features: FeatureVector,
    ) -> LaneResult:
        return self._lane.evaluate(evaluation, ctx)


class Scorer:
    """Runs the rules, computes features, and assembles a result."""

    def __init__(
        self,
        ruleset: RuleSet,
        *,
        calendar: ElectoralCalendar | None = None,
        registers: RegisterSet | None = None,
        lanes: Sequence[DetectionLane] | None = None,
        alerts: AlertIndex | None = None,
        require_verified_citations: bool = True,
        model_version: str | None = None,
        composer: ScoreComposer | None = None,
    ) -> None:
        self.engine = RuleEngine(
            ruleset,
            calendar=calendar,
            registers=registers,
            require_verified_citations=require_verified_citations,
        )
        self.features = FeatureService(
            ruleset, calendar=calendar, registers=registers
        )
        self.lanes: list[DetectionLane] = list(
            lanes if lanes is not None else [GraphLaneAdapter(alerts)]
        )
        self.model_version = model_version
        self.composer = composer or ScoreComposer()

    def score(
        self,
        donation: Donation,
        view: PointInTimeView,
        *,
        now: datetime | None = None,
        entities: Mapping[str, Entity] | None = None,
    ) -> tuple[ScoringResult, FeatureVector]:
        """Score one donation.

        Returns the feature vector alongside the result so the caller can
        persist it. A score whose inputs were not retained cannot be
        reconstructed, explained, or checked later.
        """
        ctx = self.features.context_for(donation, view, now=now, entities=entities)
        evaluation = self.engine.evaluate(
            donation, view, now=ctx.now, entities=entities
        )
        features = self.features.compute_from_context(ctx)

        results: list[LaneResult] = []
        present = {lane.name for lane in self.lanes}
        for lane in self.lanes:
            results.append(lane.evaluate(evaluation, ctx, features))
        for missing in (l for l in Lane if l not in present):
            results.append(unavailable(missing, _NOT_OPERATING[missing]))

        return (
            self.composer.compose(
                evaluation,
                tuple(results),
                feature_set_version=self.features.version,
                model_version=self.model_version,
            ),
            features,
        )


#: Why a lane is not running, when it is not configured. Stated rather than
#: left blank because "this lane found nothing" and "this lane never ran" are
#: different claims, and only the first is evidence.
_NOT_OPERATING: dict[Lane, str] = {
    Lane.CLASSIFIER: "no trained model is loaded",
    Lane.GRAPH: "graph lane is not configured",
    Lane.ANOMALY: "anomaly lane is not configured",
    Lane.REPUTATION: (
        "external reputation lane is not operating; it accuses named parties on "
        "the strength of press coverage and is switched on only once its "
        "accuracy and defamation controls are in place"
    ),
}
