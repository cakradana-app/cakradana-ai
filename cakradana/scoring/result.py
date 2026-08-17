"""Scoring output.

One result carries two independent verdicts that are never combined.

A statutory breach is a fact with a citation. A behavioural score is an
estimate. Averaging them yields a quantity that is neither auditable as a legal
finding nor interpretable as a probability, and that cannot be defended when
the subject of it objects.

Rules that could not be evaluated appear in their own list and are always
populated. A donation with no findings and three unevaluated rules has not been
cleared; it has been partly examined, and every consumer of this structure has
to be able to tell those apart.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from cakradana.rules.engine import RuleResult


class Lane(str, Enum):
    """Sources of behavioural suspicion.

    Each is capped separately rather than pooled. Lanes are not calibrated
    against one another, and the exploratory ones can always produce more
    unusual-looking donations than anyone can review; without caps they crowd
    out higher-confidence findings, and analyst trust spent on a run of weak
    alerts does not come back.
    """

    CLASSIFIER = "classifier"
    GRAPH = "graph"
    ANOMALY = "anomaly"
    REPUTATION = "reputation"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return str(self.value)


class Band(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return str(self.value)


class ReviewStatus(str, Enum):
    """Whether an analyst has read a reason's wording, in three states.

    ``UNREVIEWED`` is not a milder ``VALIDATED``. It says nobody has looked,
    which is the state every code in this system is in, and it must never be
    reported as acceptable — the point of separating it from ``REJECTED`` is
    that the two describe different failures and neither is a pass.

    It rides on the reason itself rather than sitting in an internal object,
    because the sentence and the question of whether anybody vetted it are read
    by the same person at the same moment.
    """

    VALIDATED = "validated"
    REJECTED = "rejected"
    UNREVIEWED = "unreviewed"

    @property
    def is_acceptable(self) -> bool:
        return self is ReviewStatus.VALIDATED

    def __str__(self) -> str:  # pragma: no cover - trivial
        return str(self.value)


class Reason(BaseModel):
    """Why a score is what it is, in language an analyst can act on.

    Model internals are not reasons. "Twenty-three distinct senders in nine
    days" is something a person can check; a feature index and a weight is not,
    and neither can be put to the subject of an alert.
    """

    model_config = ConfigDict(frozen=True)

    code: str
    lane: Lane
    weight: float = Field(ge=0.0, le=1.0)
    statement: str
    #: Whether an analyst has read this wording. Defaults to unreviewed, so a
    #: reason built anywhere and never stamped reads as unvetted rather than as
    #: accepted. The safe direction is the one that cannot manufacture a review
    #: nobody performed.
    wording_review: ReviewStatus = ReviewStatus.UNREVIEWED
    #: What the same quantity looks like normally. A number with no reference
    #: point is not actionable: "23 senders" means nothing until the reader
    #: knows the usual figure is three.
    comparison: str | None = None
    #: Points at the donations, cluster, or source document behind the claim.
    evidence_ref: str | None = None


class LaneResult(BaseModel):
    """One lane's contribution, or its absence.

    An absent lane is reported rather than omitted. An analyst weighing an
    alert needs to know that the reputation lane found no match, because a
    score assembled from three lanes means something different from the same
    score assembled from four.
    """

    model_config = ConfigDict(frozen=True)

    lane: Lane
    available: bool
    contribution: int = 0
    max_contribution: int = 0
    probability: float | None = Field(default=None, ge=0.0, le=1.0)
    reasons: tuple[Reason, ...] = ()
    unavailable_reason: str | None = None


class BehaviouralScore(BaseModel):
    model_config = ConfigDict(frozen=True)

    score: int = Field(ge=0, le=100)
    band: Band
    calibrated_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    lanes: tuple[LaneResult, ...]
    reasons: tuple[Reason, ...]
    #: True when one or more lanes could not run. The score is still reported,
    #: but it was assembled from less than the full picture and must not be
    #: read as though it were complete.
    degraded: bool = False
    #: The highest score obtainable given the lanes that were available.
    attainable_max: int = 100
    #: Codes in this result whose wording no analyst has read. Carried at the
    #: top level as well as on each reason, so a reader does not have to
    #: inspect every sentence to learn that some of them are unvetted.
    unreviewed_wording: tuple[str, ...] = ()
    #: Codes an analyst read and found misleading, and which are still being
    #: emitted. Kept apart from the unreviewed ones because they are a
    #: different problem: somebody looked, and said no.
    rejected_wording: tuple[str, ...] = ()


class Versions(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str | None = None
    rule_set: str
    features: str


class ScoringResult(BaseModel):
    """The complete result for one donation."""

    model_config = ConfigDict(frozen=True)

    donation_id: str
    donation_version: int
    scored_at: datetime
    versions: Versions
    legal_findings: tuple[RuleResult, ...] = ()
    indeterminate_rules: tuple[RuleResult, ...] = ()
    behavioural: BehaviouralScore | None = None

    @property
    def has_legal_findings(self) -> bool:
        return bool(self.legal_findings)

    @property
    def is_fully_evaluated(self) -> bool:
        """Whether every applicable rule reached a conclusion.

        Absence of findings only means "compliant" when this holds. Presenting
        a partly evaluated donation as clean would report unavailable reference
        data as a clean bill of health.
        """
        return not self.indeterminate_rules
