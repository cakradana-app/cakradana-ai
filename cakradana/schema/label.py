"""Label records.

Labels are a pipeline, not a dataset. Each carries where it came from, and
sources stay distinguishable through training, evaluation, and reporting.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cakradana.schema.enums import LabelSource, LabelValue

#: Provisional weights by source, reviewed against measured reliability rather
#: than left as constants. Heuristic labels enter training at a fraction of the
#: weight of an adjudicated outcome because they are hypotheses about intent
#: inferred from structure.
DEFAULT_LABEL_WEIGHTS: dict[LabelSource, float] = {
    LabelSource.DISPUTE_OUTCOME: 1.0,
    LabelSource.ANALYST_DISPOSITION: 0.9,
    LabelSource.RECIPIENT_CONFIRMATION: 0.7,
    LabelSource.RULE_TIER2: 0.5,
    LabelSource.SYNTHETIC: 0.3,
}


class Label(BaseModel):
    """One labelling event against one version of one donation."""

    model_config = ConfigDict(frozen=True)

    label_id: str
    donation_id: str
    donation_version: int = Field(ge=1)
    value: LabelValue
    source: LabelSource
    typology: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    weight: float = Field(ge=0.0, le=1.0)
    actor: str | None = None
    created_at: datetime
    #: Later labels supersede earlier ones without deleting them, so a
    #: disposition history stays reconstructible.
    superseded_by: str | None = None
    note: str | None = None

    @model_validator(mode="after")
    def _confirmation_carries_no_risk_verdict(self) -> Label:
        """A recipient confirmation may not assert that a donation is clean.

        Confirmation establishes that the transaction occurred. That is a
        different claim from the transaction being legitimate, and the
        difference matters most exactly where the system is most useful: a
        smurfed donation is genuinely received and its recipient confirms it
        truthfully. Admitting these as negative labels would train the model
        that verified smurfing is clean.
        """
        if (
            self.source is LabelSource.RECIPIENT_CONFIRMATION
            and self.value is not LabelValue.INDETERMINATE
        ):
            raise ValueError(
                "recipient confirmation records occurrence, not a risk verdict; "
                "use an analyst disposition or an adjudicated dispute outcome "
                "to assign a risk value"
            )
        return self

    @property
    def is_human(self) -> bool:
        from cakradana.schema.enums import HUMAN_LABEL_SOURCES

        return self.source in HUMAN_LABEL_SOURCES

    @classmethod
    def default_weight_for(cls, source: LabelSource) -> float:
        return DEFAULT_LABEL_WEIGHTS[source]
