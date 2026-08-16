"""Request and response shapes for the scoring service.

The caller sends a canonical donation, not engineered features. This is the
single most consequential decision in the contract. The previous service
required fourteen pre-computed inputs that no caller could produce, because the
code that computed them existed only inside a training notebook — the two
services could not be connected at all, whatever anyone intended.

Sending the record and deriving features here also makes agreement between
training and serving structural. There is one implementation of each feature
and both paths call it, so they cannot drift apart unnoticed.

A caller that supplies a feature is refused rather than obeyed. Accepting an
override would let a caller's idea of a donor's history quietly replace the
service's own, and nothing in the output would show which had been used.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cakradana.schema.enums import (
    Channel,
    EntityType,
    TemporalPrecision,
    TransactionKind,
)


class EntityRefPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_id: str | None = None
    raw_text: str | None = None
    entity_type: EntityType = EntityType.UNKNOWN
    resolution_confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _identifiable(self) -> EntityRefPayload:
        if not self.entity_id and not self.raw_text:
            raise ValueError("entity reference needs an entity_id or raw_text")
        return self


class QualityPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    extraction_confidence_min: float | None = Field(default=None, ge=0.0, le=1.0)
    has_unresolved_entity: bool | None = None


class DonationPayload(BaseModel):
    """A canonical donation as the record-owning service holds it."""

    model_config = ConfigDict(extra="forbid")

    donation_id: str
    donation_version: int = Field(default=1, ge=1)
    sender_ref: EntityRefPayload
    receiver_ref: EntityRefPayload
    amount_idr: int = Field(gt=0)
    amount_raw: str | None = None
    occurred_at: datetime
    occurred_at_precision: TemporalPrecision = TemporalPrecision.DAY
    recorded_at: datetime
    transaction_kind: TransactionKind = TransactionKind.UNKNOWN
    channel: Channel
    electoral_context: str | None = None
    is_self_funded_declared: bool | None = None
    quality: QualityPayload | None = None

    @model_validator(mode="after")
    def _timestamps_are_usable(self) -> DonationPayload:
        if self.occurred_at.tzinfo is None or self.recorded_at.tzinfo is None:
            raise ValueError(
                "timestamps must carry a timezone; donation windows are days "
                "wide and a naive timestamp silently shifts every aggregate"
            )
        if self.recorded_at < self.occurred_at:
            raise ValueError(
                "recorded_at precedes occurred_at: the system cannot have "
                "learned of a donation before it happened"
            )
        return self


class ScoreOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    explain: bool = True
    lanes: tuple[str, ...] = ("all",)


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    donation: DonationPayload
    options: ScoreOptions = ScoreOptions()


class BatchScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    #: Bounded so that one caller cannot occupy the service indefinitely.
    donations: list[DonationPayload] = Field(min_length=1, max_length=500)
    options: ScoreOptions = ScoreOptions()


class RescoreRequest(BaseModel):
    """Re-score with an explicit reason.

    Re-scoring creates a new scoring event and never overwrites the previous
    one. An analyst who cleared an alert has to be able to see what they were
    looking at when they cleared it, and a subject contesting a score has to be
    able to see the score they are contesting.
    """

    model_config = ConfigDict(extra="forbid")

    request_id: str
    donation: DonationPayload
    reason: Literal[
        "late_arriving_data",
        "dispute_upheld",
        "rule_set_change",
        "model_change",
        "correction",
    ]
    note: str | None = None


class BatchItemResult(BaseModel):
    """One item's outcome within a batch.

    Success and failure are per item. One malformed record failing an entire
    batch is how an upload silently loses everything else in it.
    """

    model_config = ConfigDict(extra="forbid")

    donation_id: str
    ok: bool
    result: dict | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"]


class ReadyResponse(BaseModel):
    """Readiness is distinct from liveness.

    A process that is running but has no rule set loaded must not receive
    traffic. Serving with no rules would report every donation as carrying no
    findings, which reads exactly like a clean result.
    """

    model_config = ConfigDict(extra="forbid")

    ready: bool
    rule_set: str | None = None
    features: str | None = None
    model: str | None = None
    detail: str | None = None


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_version: str | None
    rule_set_version: str
    feature_set_version: str
    threshold: float | None
    #: Whether the loaded model met the bar for adding detection over the
    #: rules. Exposed so an operator can see that a shipped model was not
    #: promoted on merit.
    shipped_on_merit: bool | None = None
    lanes_available: tuple[str, ...] = ()
