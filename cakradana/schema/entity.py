"""Entity records and resolution outcomes.

The accuracy ceiling of every cumulative limit rule is set here. A donor split
across three unresolved name variants evades cumulative aggregation entirely,
which is the behaviour those rules exist to catch, so resolution quality is a
detection requirement rather than data hygiene.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cakradana.schema.enums import EntityType


class Identifier(BaseModel):
    """A strong identifier attached to an entity.

    Values are held separately from analytical data with independent access
    control; feature computation and training see a surrogate key, never the
    raw value.
    """

    model_config = ConfigDict(frozen=True)

    scheme: str
    #: Surrogate reference to the protected value. The raw identifier is never
    #: carried on this record.
    value_ref: str
    validated: bool = False
    validation_note: str | None = None


class Entity(BaseModel):
    """A resolved donating or receiving party."""

    model_config = ConfigDict(frozen=True)

    entity_id: str
    canonical_name: str
    aliases: tuple[str, ...] = ()
    entity_type: EntityType = EntityType.UNKNOWN
    identifiers: tuple[Identifier, ...] = ()
    #: ISO 3166-1 alpha-2 where known. Never inferred from a name: name-based
    #: nationality inference is unreliable and discriminatory, and a rule that
    #: cannot establish jurisdiction returns indeterminate instead.
    jurisdiction: str | None = None
    #: Reference registers this entity belongs to, by register name.
    registers: tuple[str, ...] = ()
    first_seen: datetime | None = None
    last_seen: datetime | None = None

    @model_validator(mode="after")
    def _seen_range_is_ordered(self) -> Entity:
        if self.first_seen and self.last_seen and self.last_seen < self.first_seen:
            raise ValueError("last_seen precedes first_seen")
        return self

    def has_validated_identifier(self) -> bool:
        return any(i.validated for i in self.identifiers)


class ResolutionOutcome(BaseModel):
    """Result of attempting to resolve raw text to an entity.

    Every merge records its basis, confidence, and actor, and is reversible. An
    incorrect merge attributes one person's donations to another, which is both
    a serious error in this domain and something the affected subject must be
    able to contest.
    """

    model_config = ConfigDict(frozen=True)

    query: str
    entity_id: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    #: Which resolution step produced this outcome — identifier match, exact
    #: name, normalised name, or fuzzy candidate scoring.
    basis: str
    requires_review: bool = False
    candidates: tuple[tuple[str, float], ...] = ()
    actor: str | None = None
    at: datetime | None = None

    @property
    def is_resolved(self) -> bool:
        return self.entity_id is not None and not self.requires_review
