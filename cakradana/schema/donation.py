"""Canonical donation record.

Two properties of this schema carry most of its weight.

``occurred_at`` and ``recorded_at`` are separate. Point-in-time feature
computation depends on knowing what the system knew *when*, and a single date
column cannot express that. A donation that occurred in January but was
scraped in June was not knowable in February.

Absence is representable. An unresolved entity, an unparseable amount, and a
missing date are each recorded explicitly and distinguishably. Nothing is
filled with a default to make a record complete.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cakradana.schema.enums import (
    Channel,
    EntityType,
    Provenance,
    TemporalPrecision,
    TransactionKind,
)


class FieldProvenance(BaseModel):
    """How one field's current value arose."""

    model_config = ConfigDict(frozen=True)

    provenance: Provenance
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    extractor_version: str | None = None
    actor: str | None = None
    at: datetime | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def _corrections_are_attributed(self) -> FieldProvenance:
        if self.provenance is Provenance.HUMAN_CORRECTED and not self.actor:
            raise ValueError("human-corrected fields require an actor")
        return self


class SourceDocument(BaseModel):
    """Link from a donation back to the artefact it was read from.

    Without this a subject cannot meaningfully contest an attribution: they
    can be told what the system believes but not shown where it got it.
    """

    model_config = ConfigDict(frozen=True)

    kind: str
    reference: str
    retrieved_at: datetime | None = None
    #: Region of the source the value was read from — a bounding box for a
    #: scanned form, a selector or offset for a scraped page.
    locator: str | None = None
    page: int | None = None


class EntityRef(BaseModel):
    """Reference to an entity, resolved or not.

    An unresolved reference keeps its raw observed text. Rules that aggregate
    by donor refuse to evaluate against an unresolved reference rather than
    treating the raw string as an identity, because name-level identity is
    exactly what donation splitting defeats.
    """

    model_config = ConfigDict(frozen=True)

    entity_id: str | None = None
    raw_text: str | None = None
    entity_type: EntityType = EntityType.UNKNOWN
    resolution_confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _identifiable_somehow(self) -> EntityRef:
        if not self.entity_id and not self.raw_text:
            raise ValueError("entity ref needs either an entity_id or raw_text")
        return self

    @property
    def is_resolved(self) -> bool:
        return self.entity_id is not None

    @property
    def key(self) -> str:
        """Grouping key for aggregation.

        Only resolved references have one. Callers must check
        :attr:`is_resolved` first; raising here keeps an unresolved donor from
        silently forming its own single-member group and evading cumulative
        limits.
        """
        if self.entity_id is None:
            raise ValueError("unresolved entity has no aggregation key")
        return self.entity_id


class Donation(BaseModel):
    """One donation as the system understands it, at one version.

    Records are never edited in place. A correction produces a new version with
    an author and a reason, and the prior version stays retrievable, so a score
    can always name the record version it scored.
    """

    model_config = ConfigDict(frozen=True)

    donation_id: str
    donation_version: int = Field(default=1, ge=1)

    sender_ref: EntityRef
    receiver_ref: EntityRef

    amount_idr: int = Field(gt=0)
    #: The amount exactly as the source expressed it, retained whenever parsing
    #: was not unambiguous. Digit-grouping conventions differ and a misread
    #: separator moves the value by three orders of magnitude.
    amount_raw: str | None = None

    occurred_at: datetime
    occurred_at_precision: TemporalPrecision = TemporalPrecision.DAY
    recorded_at: datetime

    transaction_kind: TransactionKind = TransactionKind.UNKNOWN
    channel: Channel
    electoral_context: str | None = None
    is_self_funded_declared: bool | None = None

    source_document: SourceDocument | None = None
    provenance: dict[str, FieldProvenance] = Field(default_factory=dict)

    superseded_by: str | None = None
    correction_reason: str | None = None

    @field_validator("occurred_at", "recorded_at")
    @classmethod
    def _require_timezone(cls, value: datetime) -> datetime:
        """Reject naive timestamps.

        Donation windows are days wide and Indonesia spans three zones; a naive
        timestamp compared against an aware one is a silent correctness bug in
        every point-in-time aggregate.
        """
        if value.tzinfo is None:
            raise ValueError("timestamps must carry a timezone")
        return value

    @model_validator(mode="after")
    def _knowledge_follows_occurrence(self) -> Donation:
        if self.recorded_at < self.occurred_at:
            raise ValueError(
                "recorded_at precedes occurred_at: the system cannot have "
                "learned of a donation before it happened"
            )
        return self

    @property
    def extraction_confidence_min(self) -> float | None:
        """Lowest field confidence on this record, or ``None`` if none carry one."""
        confidences = [
            p.confidence for p in self.provenance.values() if p.confidence is not None
        ]
        return min(confidences) if confidences else None

    @property
    def has_unresolved_entity(self) -> bool:
        return not (self.sender_ref.is_resolved and self.receiver_ref.is_resolved)

    def dedup_key(self) -> tuple[Any, ...]:
        """Deterministic key for identifying the same donation seen twice.

        Truncates the timestamp to its stated precision so that the same
        donation read from a scanned form at day precision and from a digital
        submission at second precision collides rather than double-counting.
        Double-counting inflates cumulative totals and can manufacture a false
        statutory finding.
        """
        precision = self.occurred_at_precision
        stamp = self.occurred_at
        if precision is TemporalPrecision.DAY:
            stamp = stamp.replace(hour=0, minute=0, second=0, microsecond=0)
        elif precision is TemporalPrecision.HOUR:
            stamp = stamp.replace(minute=0, second=0, microsecond=0)
        elif precision is TemporalPrecision.MINUTE:
            stamp = stamp.replace(second=0, microsecond=0)
        else:
            stamp = stamp.replace(microsecond=0)

        return (
            self.sender_ref.entity_id or f"raw:{self.sender_ref.raw_text}",
            self.receiver_ref.entity_id or f"raw:{self.receiver_ref.raw_text}",
            self.amount_idr,
            stamp.isoformat(),
            self.electoral_context,
        )
