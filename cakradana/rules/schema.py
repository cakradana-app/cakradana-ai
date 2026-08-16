"""Rule definitions.

Rules are versioned data, not code. Statutory limits change with electoral
cycles, and amending a threshold must not require a code change, a redeploy,
or a model retrain.

Every rule declares the fields it needs and what to do when one is missing.
Together these implement the prohibition on silent passes: a rule that cannot
be evaluated returns indeterminate, never pass. Several statutory rules depend
on reference data that may be absent, and an unevaluated prohibition reported
as a clean result is worse than no evaluation at all, because it is
indistinguishable from one.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cakradana.schema.enums import EntityType, Regime, RuleOutcome


class LegalBasis(BaseModel):
    """Statutory grounding for a Tier-1 rule.

    Tier-2 rules carry no legal basis and must not fabricate one: they are
    behavioural heuristics, and presenting one as a legal finding would
    misstate its standing.
    """

    model_config = ConfigDict(frozen=True)

    statute: str
    article: str
    threshold_idr: int | None = None
    period: Literal["annual", "campaign", "none"] = "none"
    #: Recorded when a qualified reviewer has verified the citation and
    #: threshold against the consolidated text. A Tier-1 rule that has not
    #: been verified is refused activation by the engine.
    verified_by: str | None = None
    verified_at: date | None = None


class EffectiveWindow(BaseModel):
    """Dates between which a rule is in force.

    A donation is evaluated against the rules in force on its own date, never
    against today's. Re-scoring a 2024 donation in 2027 must apply the 2024
    rule set, or historical scores stop being reproducible.
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    from_: date = Field(alias="from")
    to: date | None = None

    def covers(self, when: date) -> bool:
        if when < self.from_:
            return False
        return self.to is None or when <= self.to


class Applicability(BaseModel):
    """Which donations a rule considers at all.

    A rule that does not apply yields ``NOT_APPLICABLE``, which is distinct
    from both ``PASS`` and ``INDETERMINATE``. Conflating "this rule has nothing
    to say" with "this rule found nothing wrong" would overstate coverage.
    """

    model_config = ConfigDict(frozen=True)

    sender_type: tuple[EntityType, ...] = ()
    receiver_type: tuple[EntityType, ...] = ()
    regime: Regime | None = None

    def applies(
        self,
        sender_type: EntityType,
        receiver_type: EntityType,
        regime: Regime,
    ) -> bool:
        if self.sender_type and sender_type not in self.sender_type:
            return False
        if self.receiver_type and receiver_type not in self.receiver_type:
            return False
        if self.regime is not None and regime is not self.regime:
            return False
        return True


class RuleTest(BaseModel):
    """The test a rule performs.

    ``kind`` selects a registered, typed implementation rather than an
    expression to interpret. Rules stay data, but the space of tests stays
    reviewable: a new kind is a code change with tests, whereas an expression
    language would let an unreviewed string change what counts as a statutory
    violation.
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    kind: str

    def params(self) -> dict[str, Any]:
        extra = self.model_extra or {}
        return dict(extra)


class RuleOutcomeSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    result: RuleOutcome
    severity: str | None = None
    reason_template: str = ""
    #: Weight a Tier-2 signal carries as a training label. This is what makes
    #: these weak supervision rather than ground truth: a heuristic label
    #: enters training at reduced weight relative to human judgement,
    #: reflecting that it is a hypothesis about intent inferred from structure.
    label_weight: float | None = Field(default=None, ge=0.0, le=1.0)


class Calibration(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["PROVISIONAL", "CALIBRATED", "RETIRED"] = "PROVISIONAL"
    review_after: str | None = None


class Rule(BaseModel):
    """One rule."""

    model_config = ConfigDict(frozen=True)

    id: str
    tier: Literal[1, 2]
    title: str
    typology: str | None = None
    legal_basis: LegalBasis | None = None
    effective: EffectiveWindow
    applies_when: Applicability = Applicability()
    test: RuleTest
    outcome: RuleOutcomeSpec
    requires_fields: tuple[str, ...] = ()
    on_missing_field: RuleOutcome = RuleOutcome.INDETERMINATE
    calibration: Calibration | None = None
    #: Set when a rule is specified but must not fire, because the data it
    #: depends on does not exist yet. An inactive rule reports indeterminate
    #: rather than disappearing, so its absence stays visible.
    active: bool = True
    inactive_reason: str | None = None

    @model_validator(mode="after")
    def _tier_one_is_a_legal_finding(self) -> Rule:
        if self.tier == 1:
            if self.legal_basis is None:
                raise ValueError(f"{self.id}: a statutory rule requires a legal basis")
            if self.outcome.result is not RuleOutcome.LEGAL_FINDING:
                raise ValueError(
                    f"{self.id}: a statutory rule must produce a legal finding"
                )
            if self.outcome.label_weight is not None:
                raise ValueError(
                    f"{self.id}: statutory findings are not training labels"
                )
        return self

    @model_validator(mode="after")
    def _tier_two_claims_no_legal_authority(self) -> Rule:
        if self.tier == 2:
            if self.legal_basis is not None:
                raise ValueError(
                    f"{self.id}: behavioural heuristics have no legal basis and "
                    f"must not cite one"
                )
            if self.outcome.result is not RuleOutcome.BEHAVIOURAL_SIGNAL:
                raise ValueError(
                    f"{self.id}: a behavioural heuristic must not produce a "
                    f"legal finding"
                )
        return self

    @property
    def is_verified(self) -> bool:
        """Whether a statutory rule's citation has been reviewed.

        Consistency across source documents is not verification. An unverified
        statutory rule may be loaded and inspected but is not permitted to
        produce findings against real data.
        """
        if self.tier != 1:
            return True
        return bool(self.legal_basis and self.legal_basis.verified_by)


class RuleSet(BaseModel):
    """An immutable, published collection of rules.

    A change creates a new version. Retired rules are retained rather than
    deleted so that historical evaluations stay reproducible.
    """

    model_config = ConfigDict(frozen=True)

    version: str
    rules: tuple[Rule, ...]
    notes: str | None = None
    #: True for a set that exercises rules against fixture reference data.
    #: Such a set is never selected implicitly: it produces findings that look
    #: like enforcement and are not, so choosing it has to be a deliberate act
    #: by whoever is running the system.
    demonstration: bool = False

    @model_validator(mode="after")
    def _ids_are_unique(self) -> RuleSet:
        seen: set[str] = set()
        for rule in self.rules:
            if rule.id in seen:
                raise ValueError(f"duplicate rule id {rule.id} in {self.version}")
            seen.add(rule.id)
        return self

    def in_force_on(self, when: date) -> tuple[Rule, ...]:
        return tuple(r for r in self.rules if r.effective.covers(when))

    def by_id(self, rule_id: str) -> Rule:
        for rule in self.rules:
            if rule.id == rule_id:
                return rule
        raise KeyError(rule_id)

    @property
    def tier1(self) -> tuple[Rule, ...]:
        return tuple(r for r in self.rules if r.tier == 1)

    @property
    def tier2(self) -> tuple[Rule, ...]:
        return tuple(r for r in self.rules if r.tier == 2)
