"""Composing a result from the rule engine and the detection lanes.

Two things happen here and they stay apart. Statutory findings pass through
untouched, ordered ahead of everything else, because they are the system's most
defensible output and the only part of it that asserts a fact. Behavioural
lanes are combined into a single 0–100 figure that ranks donations against each
other and claims nothing more.

Each lane has a fixed ceiling. Lanes are not calibrated against one another,
and the exploratory ones will always find more unusual donations than a team
can review, so pooling them lets the weakest evidence displace the strongest.

When a lane cannot run, its points are simply unavailable and the result says
so. Rescaling the remaining lanes to fill the gap would make a partial score
indistinguishable from a complete one, which is the same error as filling a
missing feature with zero.
"""

from __future__ import annotations

from datetime import datetime

from cakradana.rules.engine import RuleEvaluation
from cakradana.scoring.result import (
    Band,
    BehaviouralScore,
    Lane,
    LaneResult,
    Reason,
    ReviewStatus,
    ScoringResult,
    Versions,
)
from cakradana.scoring.review import default_statuses

#: Share of the behavioural score each lane may contribute. Ordered by how much
#: an analyst can rely on the evidence behind it, so the exploratory lanes
#: cannot outweigh the ones with a track record.
LANE_CEILINGS: dict[Lane, int] = {
    Lane.CLASSIFIER: 50,
    Lane.GRAPH: 30,
    Lane.ANOMALY: 15,
    Lane.REPUTATION: 5,
}

#: Band boundaries, tuned against how many alerts a team can actually review
#: rather than fixed by anything intrinsic.
BAND_BOUNDARIES: tuple[tuple[int, Band], ...] = (
    (25, Band.LOW),
    (50, Band.MODERATE),
    (75, Band.HIGH),
    (101, Band.CRITICAL),
)

MAX_REASONS_SHOWN = 5


class MissingReasons(RuntimeError):
    """Raised when a score would be surfaced with nothing to justify it.

    A bare number is unusable here: an analyst cannot triage it, a subject
    cannot contest it, and an auditor cannot review it. Withholding the score
    and raising is preferable to publishing one that cannot be explained.
    """


def band_for(score: int) -> Band:
    for boundary, band in BAND_BOUNDARIES:
        if score < boundary:
            return band
    return Band.CRITICAL


class ScoreComposer:
    """Assembles a scoring result."""

    def __init__(
        self,
        *,
        ceilings: dict[Lane, int] | None = None,
        require_reasons: bool = True,
        wording_statuses: dict[str, ReviewStatus] | None = None,
    ) -> None:
        self.ceilings = dict(ceilings or LANE_CEILINGS)
        self.require_reasons = require_reasons
        #: Review state per reason code, read from the shipped ledger unless a
        #: caller supplies its own. A code the ledger says nothing about reads
        #: unreviewed, which is what it is.
        self.wording_statuses = wording_statuses

    def compose(
        self,
        evaluation: RuleEvaluation,
        lanes: tuple[LaneResult, ...],
        *,
        feature_set_version: str,
        model_version: str | None = None,
        scored_at: datetime | None = None,
    ) -> ScoringResult:
        behavioural = self._behavioural(lanes)
        return ScoringResult(
            donation_id=evaluation.donation_id,
            donation_version=evaluation.donation_version,
            scored_at=scored_at or evaluation.evaluated_at,
            versions=Versions(
                model=model_version,
                rule_set=evaluation.rule_set_version,
                features=feature_set_version,
            ),
            legal_findings=evaluation.legal_findings,
            indeterminate_rules=evaluation.indeterminate,
            behavioural=behavioural,
        )

    def _behavioural(self, lanes: tuple[LaneResult, ...]) -> BehaviouralScore | None:
        if not lanes:
            return None

        statuses = (
            self.wording_statuses
            if self.wording_statuses is not None
            else default_statuses()
        )
        lanes = tuple(_reviewed(lane, statuses) for lane in lanes)

        score = sum(lane.contribution for lane in lanes if lane.available)
        attainable = sum(
            self.ceilings.get(lane.lane, 0) for lane in lanes if lane.available
        )
        degraded = any(not lane.available for lane in lanes)

        reasons = self._ordered_reasons(lanes, statuses)
        if self.require_reasons and score > 0 and not reasons:
            raise MissingReasons(
                "a behavioural score was produced with no reasons; the score is "
                "withheld rather than surfaced unexplained"
            )

        classifier = next(
            (lane for lane in lanes if lane.lane is Lane.CLASSIFIER and lane.available),
            None,
        )

        return BehaviouralScore(
            score=min(score, 100),
            band=band_for(min(score, 100)),
            calibrated_probability=classifier.probability if classifier else None,
            lanes=lanes,
            reasons=reasons,
            degraded=degraded,
            attainable_max=attainable,
            unreviewed_wording=_codes_at(reasons, ReviewStatus.UNREVIEWED),
            rejected_wording=_codes_at(reasons, ReviewStatus.REJECTED),
        )

    def _ordered_reasons(
        self, lanes: tuple[LaneResult, ...], statuses: dict[str, ReviewStatus]
    ) -> tuple[Reason, ...]:
        collected: list[Reason] = []
        for lane in lanes:
            if lane.available:
                collected.extend(lane.reasons)
            elif lane.unavailable_reason:
                # An absent lane is itself something the reader needs. A score
                # built from three lanes is a different claim from the same
                # score built from four.
                collected.append(
                    Reason(
                        code="LANE_UNAVAILABLE",
                        lane=lane.lane,
                        weight=0.0,
                        statement=(
                            f"The {lane.lane} lane did not run: "
                            f"{lane.unavailable_reason}."
                        ),
                        wording_review=statuses.get(
                            "LANE_UNAVAILABLE", ReviewStatus.UNREVIEWED
                        ),
                    )
                )
        collected.sort(key=lambda r: r.weight, reverse=True)
        return tuple(collected)


def _reviewed(
    lane: LaneResult, statuses: dict[str, ReviewStatus]
) -> LaneResult:
    """Stamp each of a lane's reasons with whether anybody vetted its wording.

    Done here rather than in the lanes so that there is one place a reason can
    acquire a review state, and so that a lane cannot claim one for itself. A
    code the ledger does not mention keeps the unreviewed default, which is
    also what an uncatalogued code gets: nobody has read it either.
    """
    if not lane.reasons:
        return lane
    return lane.model_copy(
        update={
            "reasons": tuple(
                reason.model_copy(
                    update={
                        "wording_review": statuses.get(
                            reason.code, ReviewStatus.UNREVIEWED
                        )
                    }
                )
                for reason in lane.reasons
            )
        }
    )


def _codes_at(reasons: tuple[Reason, ...], status: ReviewStatus) -> tuple[str, ...]:
    return tuple(sorted({r.code for r in reasons if r.wording_review is status}))


def unavailable(lane: Lane, reason: str) -> LaneResult:
    return LaneResult(
        lane=lane,
        available=False,
        max_contribution=LANE_CEILINGS.get(lane, 0),
        unavailable_reason=reason,
    )


def contribution_from(
    lane: Lane,
    intensity: float,
    reasons: tuple[Reason, ...],
    *,
    probability: float | None = None,
    ceilings: dict[Lane, int] | None = None,
) -> LaneResult:
    """Scale a lane's 0–1 intensity onto its share of the score."""
    ceiling = (ceilings or LANE_CEILINGS).get(lane, 0)
    intensity = min(max(intensity, 0.0), 1.0)
    return LaneResult(
        lane=lane,
        available=True,
        contribution=round(intensity * ceiling),
        max_contribution=ceiling,
        probability=probability,
        reasons=reasons,
    )
