"""The scoring service, independent of any web framework.

Holds the rule engine, the feature service, the lanes, and the point-in-time
state features are derived from. Keeping it separate from the HTTP layer means
the behaviour that matters — what is refused, what is degraded, what versions a
result carries — is testable without a client.

Derived state is rebuildable by replaying canonical records from the service
that owns them. This one holds no canonical data of its own: it computes over a
copy and can always be rebuilt from the source of truth, which is what keeps a
drifting cache from silently producing features nobody can reproduce.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, Mapping, Sequence

from cakradana.calendar import ElectoralCalendar
from cakradana.history import InMemoryDonationStore
from cakradana.lanes.alerts import AlertIndex, GroupAlert, GroupAlertDetector
from cakradana.lanes.classifier import ClassifierLane
from cakradana.registers import RegisterSet
from cakradana.rules import RuleSet, load_latest
from cakradana.scoring.result import Lane, ScoringResult
from cakradana.scoring.scorer import Scorer, GraphLaneAdapter
from cakradana.schema import Donation, Entity
from cakradana.serving.schemas import DonationPayload
from cakradana.training.registry import Artifact


class ServiceNotReady(RuntimeError):
    """Raised when the service is asked to score before it can."""


class ScoringService:
    """Scores donations and maintains the history features derive from."""

    def __init__(
        self,
        *,
        ruleset: RuleSet | None = None,
        calendar: ElectoralCalendar | None = None,
        registers: RegisterSet | None = None,
        artifact: Artifact | None = None,
        entities: Mapping[str, Entity] | None = None,
        detector: GroupAlertDetector | None = None,
        require_verified_citations: bool = True,
    ) -> None:
        self.ruleset = ruleset or load_latest()
        self.artifact = artifact
        self.entities: dict[str, Entity] = dict(entities or {})
        self.store = InMemoryDonationStore()

        self.detector = detector or GroupAlertDetector()
        self._graph = GraphLaneAdapter()
        lanes = [self._graph]
        if artifact is not None:
            lanes.append(ClassifierLane(artifact))

        self.scorer = Scorer(
            self.ruleset,
            calendar=calendar,
            registers=registers,
            lanes=lanes,
            require_verified_citations=require_verified_citations,
            model_version=artifact.version if artifact else None,
        )
        #: Scoring events, newest last, keyed by donation. Re-scoring appends
        #: rather than replacing, so the history of what was said about a
        #: donation stays intact.
        self._events: dict[str, list[ScoringResult]] = {}
        #: When structural detection last ran. Surfaced with the alerts,
        #: because "detection found nothing" and "detection has not run" are
        #: different claims and an empty list is not enough to tell them apart.
        self.alerts_detected_at: datetime | None = None

    # -- readiness -------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        return bool(self.ruleset and self.ruleset.rules)

    def readiness_detail(self) -> str | None:
        if not self.ruleset or not self.ruleset.rules:
            return (
                "no rule set is loaded; serving without rules would report "
                "every donation as carrying no findings, which is "
                "indistinguishable from a clean result"
            )
        return None

    # -- state -----------------------------------------------------------

    def replay(
        self,
        donations: Iterable[Donation],
        *,
        entities: Mapping[str, Entity] | None = None,
    ) -> int:
        """Rebuild derived state from canonical records."""
        self.store = InMemoryDonationStore(donations)
        if entities:
            self.entities.update(entities)
        return len(self.store)

    def remember(self, donation: Donation) -> None:
        """Add a donation to the history later donations are judged against."""
        self.store.add(donation)

    # -- scoring ---------------------------------------------------------

    def score(
        self,
        payload: DonationPayload,
        *,
        now: datetime | None = None,
        remember: bool = True,
    ) -> ScoringResult:
        if not self.is_ready:
            raise ServiceNotReady(self.readiness_detail() or "service is not ready")

        donation = to_donation(payload)
        if remember:
            # Recorded before scoring so that the donation participates in its
            # own cumulative total. A limit breach attaches to a donation that
            # is itself part of the excess.
            self.remember(donation)

        view = self.store.knowable_at(donation.occurred_at)
        result, _features = self.scorer.score(
            donation, view, now=now, entities=self.entities
        )
        self._events.setdefault(donation.donation_id, []).append(result)
        return result

    def score_many(
        self, payloads: Sequence[DonationPayload], *, now: datetime | None = None
    ) -> list[tuple[DonationPayload, ScoringResult | None, str | None]]:
        """Score a batch, per item.

        A failing item yields its own error and the rest are scored. One
        malformed record failing the whole batch is how an upload quietly loses
        everything else it contained.
        """
        outcomes: list[tuple[DonationPayload, ScoringResult | None, str | None]] = []
        for payload in payloads:
            try:
                outcomes.append((payload, self.score(payload, now=now), None))
            except Exception as error:  # noqa: BLE001 - reported per item
                outcomes.append((payload, None, str(error)))
        return outcomes

    # -- group alerts ----------------------------------------------------

    def detect_group_alerts(
        self, *, as_of: datetime | None = None
    ) -> tuple[GroupAlert, ...]:
        """Find structural patterns across the population.

        Runs over everything knowable at `as_of` rather than per donation,
        because a cluster is not a property of any of its members. The result
        is adopted by the graph lane, so donations scored after this call carry
        the cluster's evidence; ones scored before it keep what they were given
        until they are explicitly rescored.
        """
        moment = as_of or self.store.latest_recorded_at
        if moment is None:
            self._graph.use(AlertIndex())
            self.alerts_detected_at = None
            return ()

        alerts = self.detector.detect(self.store.knowable_at(moment), as_of=moment)
        self._graph.use(AlertIndex(alerts))
        self.alerts_detected_at = moment
        return alerts

    @property
    def group_alerts(self) -> tuple[GroupAlert, ...]:
        return tuple(self._graph.alerts)

    def history_for(self, donation_id: str) -> tuple[ScoringResult, ...]:
        return tuple(self._events.get(donation_id, ()))

    def latest_for(self, donation_id: str) -> ScoringResult | None:
        events = self._events.get(donation_id)
        return events[-1] if events else None

    # -- introspection ---------------------------------------------------

    @property
    def available_lanes(self) -> tuple[str, ...]:
        return tuple(str(lane.name) for lane in self.scorer.lanes)

    @property
    def feature_set_version(self) -> str:
        return self.scorer.features.version

    @property
    def threshold(self) -> float | None:
        return self.artifact.threshold if self.artifact else None


def to_donation(payload: DonationPayload) -> Donation:
    """Convert a request payload into a canonical record.

    Quality figures supplied by the caller are not carried onto the record.
    They describe the caller's own extraction and would otherwise be
    indistinguishable from provenance this service established for itself.
    """
    from cakradana.schema import EntityRef

    return Donation(
        donation_id=payload.donation_id,
        donation_version=payload.donation_version,
        sender_ref=EntityRef(
            entity_id=payload.sender_ref.entity_id,
            raw_text=payload.sender_ref.raw_text,
            entity_type=payload.sender_ref.entity_type,
            resolution_confidence=payload.sender_ref.resolution_confidence,
        ),
        receiver_ref=EntityRef(
            entity_id=payload.receiver_ref.entity_id,
            raw_text=payload.receiver_ref.raw_text,
            entity_type=payload.receiver_ref.entity_type,
            resolution_confidence=payload.receiver_ref.resolution_confidence,
        ),
        amount_idr=payload.amount_idr,
        amount_raw=payload.amount_raw,
        occurred_at=payload.occurred_at,
        occurred_at_precision=payload.occurred_at_precision,
        recorded_at=payload.recorded_at,
        transaction_kind=payload.transaction_kind,
        channel=payload.channel,
        electoral_context=payload.electoral_context,
        is_self_funded_declared=payload.is_self_funded_declared,
    )
