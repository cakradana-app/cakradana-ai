"""Rule evaluation.

Evaluation produces two independent outputs that are never combined: statutory
findings, which are facts with citations, and behavioural signals, which are
hypotheses. A donation can carry both, and each is reported on its own terms.

Rules that could not be evaluated are reported too. A donation with no findings
and three indeterminate rules has not been cleared — it has been partially
evaluated, and anything consuming this output has to be able to tell the
difference.
"""

from __future__ import annotations

from datetime import date, datetime
from string import Formatter
from typing import Iterable, Mapping

from pydantic import BaseModel, ConfigDict

from cakradana.calendar import ElectoralCalendar
from cakradana.history import PointInTimeView
from cakradana.registers import RegisterSet, empty_register_set
from cakradana.reporting import SubmissionSet, no_submissions
from cakradana.rules.context import LimitTable, RuleContext, rule_applies
from cakradana.rules.predicates import PredicateResult, get_predicate
from cakradana.rules.schema import Rule, RuleSet
from cakradana.schema import Donation, Entity
from cakradana.schema.enums import Regime, RuleOutcome


class RuleResult(BaseModel):
    """What one rule concluded about one donation."""

    model_config = ConfigDict(frozen=True)

    rule_id: str
    tier: int
    outcome: RuleOutcome
    typology: str | None = None
    severity: str | None = None
    statute: str | None = None
    article: str | None = None
    threshold_idr: int | None = None
    observed: float | int | None = None
    explanation: str | None = None
    #: Why the rule could not be evaluated, when it could not be.
    reason: str | None = None
    label_weight: float | None = None
    facts: Mapping[str, object] = {}
    skipped_conditions: tuple[str, ...] = ()
    #: False when the evidence behind this finding is not authoritative — a
    #: fixture register standing in for one nobody has supplied. Such a
    #: finding demonstrates that the rule works; it does not establish that
    #: the donor did anything.
    authoritative: bool = True

    @property
    def is_demonstration(self) -> bool:
        return self.outcome is RuleOutcome.LEGAL_FINDING and not self.authoritative


class RuleEvaluation(BaseModel):
    """The full result of running a rule set against one donation."""

    model_config = ConfigDict(frozen=True)

    donation_id: str
    donation_version: int
    rule_set_version: str
    evaluated_at: datetime
    results: tuple[RuleResult, ...]

    @property
    def legal_findings(self) -> tuple[RuleResult, ...]:
        return tuple(
            r for r in self.results if r.outcome is RuleOutcome.LEGAL_FINDING
        )

    @property
    def behavioural_signals(self) -> tuple[RuleResult, ...]:
        return tuple(
            r for r in self.results if r.outcome is RuleOutcome.BEHAVIOURAL_SIGNAL
        )

    @property
    def indeterminate(self) -> tuple[RuleResult, ...]:
        return tuple(
            r for r in self.results if r.outcome is RuleOutcome.INDETERMINATE
        )

    @property
    def is_fully_evaluated(self) -> bool:
        """Whether every applicable rule reached a conclusion.

        The absence of findings only means something when this is true.
        """
        return not self.indeterminate

    def tier2_label_weight(self) -> float:
        """Combined weight of the behavioural signals that fired.

        Used as weak supervision. Returns the strongest single signal rather
        than a sum: several heuristics firing on the same underlying structure
        is common and adding them would manufacture confidence out of
        correlated evidence.
        """
        weights = [
            r.label_weight for r in self.behavioural_signals if r.label_weight
        ]
        return max(weights) if weights else 0.0


class RuleEngine:
    """Evaluates a rule set against donations.

    Rules in force are selected by the donation's own date, so re-scoring an
    older donation applies the rules that governed it rather than today's.
    """

    def __init__(
        self,
        ruleset: RuleSet,
        *,
        calendar: ElectoralCalendar | None = None,
        registers: RegisterSet | None = None,
        submissions: SubmissionSet | None = None,
        require_verified_citations: bool = True,
        annual_period_start_month: int = 1,
    ) -> None:
        self.ruleset = ruleset
        self.calendar = calendar or ElectoralCalendar()
        self.registers = registers or empty_register_set()
        self.submissions = submissions or no_submissions()
        self.limits = LimitTable.from_ruleset(ruleset)
        self.require_verified_citations = require_verified_citations
        self.annual_period_start_month = annual_period_start_month

    def evaluate(
        self,
        donation: Donation,
        view: PointInTimeView,
        *,
        now: datetime | None = None,
        entities: Mapping[str, Entity] | None = None,
    ) -> RuleEvaluation:
        now = now or datetime.now(tz=donation.occurred_at.tzinfo)
        ctx = RuleContext(
            donation=donation,
            view=view,
            calendar=self.calendar,
            registers=self.registers,
            limits=self.limits,
            now=now,
            entities=entities or {},
            submissions=self.submissions,
            annual_period_start_month=self.annual_period_start_month,
        )

        results = [
            self._evaluate_rule(rule, ctx)
            for rule in self.ruleset.in_force_on(donation.occurred_at.date())
        ]

        return RuleEvaluation(
            donation_id=donation.donation_id,
            donation_version=donation.donation_version,
            rule_set_version=self.ruleset.version,
            evaluated_at=now,
            results=tuple(results),
        )

    def _evaluate_rule(self, rule: Rule, ctx: RuleContext) -> RuleResult:
        if not rule.active:
            return self._undetermined(
                rule,
                rule.inactive_reason or "rule is specified but not yet operable",
            )

        if (
            rule.tier == 1
            and self.require_verified_citations
            and not rule.is_verified
        ):
            # Consistency across source documents is not verification. A
            # statutory finding is asserted as fact and cited by article, so an
            # unreviewed citation must not reach a subject.
            return self._undetermined(
                rule,
                "statutory citation has not been verified by a qualified reviewer",
            )

        if rule.applies_when.regime is not None and ctx.regime is Regime.INDETERMINATE:
            # The rule is scoped to a limit regime and the regime could not be
            # established, so whether it applies is itself unknown. Reporting
            # "not applicable" here would be the silent pass this engine exists
            # to prevent: it reads as the rule having been considered and found
            # irrelevant, when in fact no limit was checked at all.
            return self._undetermined(
                rule,
                "the applicable limit regime could not be determined, so it is "
                "not known whether this rule governs the donation",
            )

        if not rule_applies(rule, ctx):
            return RuleResult(
                rule_id=rule.id,
                tier=rule.tier,
                outcome=RuleOutcome.NOT_APPLICABLE,
                typology=rule.typology,
            )

        missing = ctx.missing_fields(rule.requires_fields)
        if missing:
            return RuleResult(
                rule_id=rule.id,
                tier=rule.tier,
                outcome=rule.on_missing_field,
                typology=rule.typology,
                reason=f"required data unavailable: {', '.join(missing)}",
            )

        outcome = get_predicate(rule.test.kind)(rule, ctx)
        if outcome.indeterminate:
            return self._undetermined(rule, outcome.indeterminate)
        if not outcome.fired:
            return RuleResult(
                rule_id=rule.id,
                tier=rule.tier,
                outcome=RuleOutcome.PASS,
                typology=rule.typology,
                observed=outcome.observed,
                facts=dict(outcome.facts),
                skipped_conditions=outcome.skipped,
            )

        return self._finding(rule, ctx, outcome)

    def _finding(
        self, rule: Rule, ctx: RuleContext, outcome: PredicateResult
    ) -> RuleResult:
        basis = rule.legal_basis
        return RuleResult(
            rule_id=rule.id,
            tier=rule.tier,
            outcome=rule.outcome.result,
            typology=rule.typology,
            severity=rule.outcome.severity,
            statute=basis.statute if basis else None,
            article=basis.article if basis else None,
            threshold_idr=(
                int(outcome.threshold)
                if outcome.threshold is not None
                else (basis.threshold_idr if basis else None)
            ),
            observed=outcome.observed,
            explanation=_explain(rule, ctx, outcome),
            label_weight=rule.outcome.label_weight,
            facts=dict(outcome.facts),
            skipped_conditions=outcome.skipped,
            authoritative=outcome.authoritative,
        )

    @staticmethod
    def _undetermined(rule: Rule, reason: str) -> RuleResult:
        return RuleResult(
            rule_id=rule.id,
            tier=rule.tier,
            outcome=RuleOutcome.INDETERMINATE,
            typology=rule.typology,
            reason=reason,
        )


def _explain(rule: Rule, ctx: RuleContext, outcome: PredicateResult) -> str:
    """Render the explanation, marking any that rests on non-authoritative data.

    The qualifier leads rather than trails. A reader who stops after the first
    clause must not come away believing an offence has been established.
    """
    rendered = _render(rule, ctx, outcome)
    if outcome.authoritative:
        return rendered
    return (
        "DEMONSTRATION ONLY — this finding rests on fixture reference data, "
        "not on the authoritative register, and establishes nothing about the "
        f"donor: {rendered}"
    )


def _render(rule: Rule, ctx: RuleContext, outcome: PredicateResult) -> str:
    """Fill a reason template from the donation and the test's own findings.

    A missing placeholder leaves a visible marker rather than raising. An
    explanation is what an analyst acts on and what a subject is shown, so a
    template defect must degrade into something legible instead of suppressing
    an otherwise valid finding.
    """
    template = rule.outcome.reason_template
    if not template:
        return ""

    donation = ctx.donation
    basis = rule.legal_basis
    values: dict[str, object] = {
        "amount": _rupiah(donation.amount_idr),
        "date": donation.occurred_at.date().isoformat(),
        "sender": donation.sender_ref.raw_text or donation.sender_ref.entity_id or "?",
        "receiver": (
            donation.receiver_ref.raw_text or donation.receiver_ref.entity_id or "?"
        ),
        "statute": basis.statute if basis else "",
        "article": basis.article if basis else "",
    }
    for key, value in outcome.facts.items():
        values[key] = _rupiah(value) if key in _MONEY_FACTS else value
    if outcome.threshold is not None:
        values["threshold"] = _rupiah(outcome.threshold)

    rendered: list[str] = []
    for literal, field_name, _spec, _conv in Formatter().parse(template):
        rendered.append(literal)
        if field_name is None:
            continue
        rendered.append(str(values.get(field_name, f"<{field_name}?>")))
    return " ".join("".join(rendered).split())


_MONEY_FACTS = frozenset({"total", "prior_total", "threshold", "inflow_total",
                          "outflow", "recipient_total", "from_this_donor"})


def _rupiah(amount: object) -> str:
    if isinstance(amount, (int, float)):
        return f"Rp{int(amount):,}".replace(",", ".")
    return str(amount)


def rules_in_force(ruleset: RuleSet, when: date) -> Iterable[Rule]:
    return ruleset.in_force_on(when)
