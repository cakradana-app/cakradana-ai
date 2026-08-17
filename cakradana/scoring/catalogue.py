"""Every reason code the system can put in front of an analyst.

A reason is the part of a score a person actually reads. It is what an analyst
triages on, what a subject is shown when they ask why they were flagged, and
what an auditor reads back afterwards. Until this module existed the set of
codes was implicit — scattered across four lanes and the composer — so nobody
could enumerate what an analyst would be reviewing, and therefore nobody could
review it.

Two things are enumerated here and they are different.

**What can be emitted.** Every code, the lanes that emit it, the observation it
states, and the wording templates it states it in. Codes whose domain is
defined elsewhere — the behavioural rule set, the group-alert kinds, the feature
set — are derived from that definition rather than copied, because a copy drifts
and a catalogue that has drifted reads as a guarantee while being wrong.

**Whether anybody has read it.** That lives in the review ledger, not here: it
changes when a person makes a decision, and the code it describes does not.

The wording rules are checkable and are checked. A reason states an observation
— "twenty-three distinct senders in nine days" — not a conclusion about what the
observation means, and never a model internal. A feature index and a weight are
properties of the model rather than of the donation, and neither can be put to
the person the alert is about.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from cakradana.scoring.result import Lane

#: How a value is put into a sentence. Part of the wording rather than a
#: presentation detail: a rupiah figure read under the wrong digit-grouping
#: convention is wrong by three orders of magnitude, and a share printed as
#: ``0.42`` is a different claim from one printed as ``42%``.
Render = Literal["plain", "rupiah", "share", "when_true"]


class ReasonCode(BaseModel):
    """One code the system can emit, and the wording it emits it in."""

    model_config = ConfigDict(frozen=True)

    code: str
    #: Every lane that can emit this code. More than one only for the codes the
    #: composer writes on a lane's behalf.
    lanes: tuple[Lane, ...]
    #: The rule, alert kind, detector, or feature the wording comes from. An
    #: analyst who disputes a statement has to be able to reach what produced
    #: it, and a code that names no source cannot be traced back to anything.
    source: str
    #: What the code states, in the catalogue's own words. One line, so that a
    #: reviewer can read the whole catalogue.
    observation: str
    #: The wording templates actually emitted, with ``{placeholders}`` where
    #: values are substituted. This is the text under review — not the summary
    #: above it. Empty exactly when there is no wording, because the quantity
    #: has no form a reader could check.
    statements: tuple[str, ...] = ()
    #: What the same quantity usually looks like, or the reference the figure
    #: should be read against. A number with no reference point is not
    #: actionable.
    comparison: str | None = None
    render: Render = "plain"
    #: Whether this code may be stated to an analyst or to the subject of an
    #: alert. False marks a derived quantity whose definition lives only in the
    #: feature code — a coordinate in the model's input space rather than
    #: something a person holding the records could verify. Such a code is
    #: catalogued so the enumeration stays complete, and is refused a place in
    #: a reason so that it cannot reach a case bundle.
    analyst_facing: bool = True

    @model_validator(mode="after")
    def _stateable_iff_facing(self) -> ReasonCode:
        if self.analyst_facing and not self.statements:
            raise ValueError(
                f"{self.code} is marked as something an analyst may be shown "
                f"but carries no wording to show them"
            )
        if not self.analyst_facing and self.statements:
            raise ValueError(
                f"{self.code} carries wording while being marked as never "
                f"shown; wording that reaches nobody is not under review, and "
                f"leaving it here would put it back in front of somebody"
            )
        return self

    def matches(self, statement: str) -> bool:
        """Whether an emitted statement was produced by one of the templates.

        A qualifier may be appended after the sentence — the group lanes add
        one when part of a pattern rests on unresolved parties — so a trailing
        clause is allowed. Anything else means the wording in the code and the
        wording in the catalogue have parted company.
        """
        return any(
            re.fullmatch(_as_pattern(template), statement, flags=re.DOTALL)
            for template in self.statements
        )


def _as_pattern(template: str) -> str:
    """Turn a wording template into a pattern that matches what it renders.

    Placeholders become a non-greedy any-run; everything else is matched
    literally, because a template full of punctuation is otherwise a pattern
    full of metacharacters.
    """
    parts = re.split(r"\{[^{}]*\}", template)
    body = ".+?".join(re.escape(part) for part in parts)
    return body + r"(?:\s.*)?"


# ---------------------------------------------------------------------------
# The catalogue
# ---------------------------------------------------------------------------

#: Codes with a wording written in one place. The rest are derived below from
#: whatever defines their domain.
_FIXED: tuple[ReasonCode, ...] = (
    ReasonCode(
        code="LANE_UNAVAILABLE",
        lanes=tuple(Lane),
        source="score composition",
        observation="a lane did not run, and why",
        statements=("The {lane} lane did not run: {reason}.",),
    ),
    ReasonCode(
        code="LAYERING_CHAIN",
        lanes=(Lane.GRAPH,),
        source="group alert: layering chain",
        observation=(
            "this donation is one of several that occurred in sequence between "
            "the same parties inside one window"
        ),
        statements=(
            "This donation is one of {donations} that occurred in sequence "
            "between {counterparties} other parties within {span} days.",
        ),
    ),
    ReasonCode(
        code="UNUSUAL_COMBINATION",
        lanes=(Lane.ANOMALY,),
        source="anomaly detector",
        observation=(
            "the donation fell outside the range this lane was fitted to treat "
            "as ordinary, with no attribution to any quantity"
        ),
        statements=(
            "The anomaly lane placed this donation outside the range it was "
            "fitted to treat as ordinary. It does not identify which "
            "quantities put it there, so nothing here can be checked against "
            "the record.",
        ),
    ),
    ReasonCode(
        code="ADVERSE_COVERAGE",
        lanes=(Lane.REPUTATION,),
        source="adverse coverage index",
        observation=(
            "published coverage naming this donor exists, at a stated stage of "
            "proceedings"
        ),
        statements=(
            "{count} item(s) of published coverage across {sources} "
            "independent sources report {stage} concerning this donor. This is "
            "what has been written, not a finding about what the donor did.",
        ),
    ),
    ReasonCode(
        code="MODEL_SCORE",
        lanes=(Lane.CLASSIFIER,),
        source="classifier, with no rankable input available",
        observation=(
            "the classifier contributed, and none of the inputs it relied on "
            "most carries wording a reader can check"
        ),
        statements=(
            "The classifier lane contributed to this donation's assessment, "
            "but none of the inputs it relied on most carries wording a "
            "reader can check, so nothing here can be checked against the "
            "record.",
        ),
    ),
)

#: Wording for the group alerts whose shape the graph lane restates. Each also
#: has a per-donation form supplied by the rule set, added alongside it below.
_ALERT_STATEMENTS: dict[str, str] = {
    "FAN_IN_BURST": (
        "This donation is one of {donations} reaching the same recipient from "
        "{counterparties} distinct donors within {days} days."
    ),
    "FAN_OUT": (
        "This donation is one of {donations} from the same donor to "
        "{counterparties} distinct recipients within {days} days."
    ),
}

_ALERT_OBSERVATIONS: dict[str, str] = {
    "FAN_IN_BURST": "many distinct donors reached one recipient in a short window",
    "FAN_OUT": "one donor reached many distinct recipients in a short window",
    "PASS_THROUGH": (
        "a donor received a comparable amount shortly before giving it onward"
    ),
    "DONOR_CONCENTRATION": (
        "one donor supplied most of a recipient's recorded funding"
    ),
}


def _graph_codes() -> tuple[ReasonCode, ...]:
    """Codes the graph lane emits, worded by the rule set and the alert kinds.

    The per-donation wording is the behavioural rule's own reason template, read
    from the rule set rather than restated here. Rules are data in this system,
    and a rule whose wording is amended in the YAML must not leave a stale copy
    of the old wording sitting in a catalogue that claims to be complete.
    """
    from cakradana.lanes.graph import STRUCTURAL_RULES
    from cakradana.rules import load_latest

    templates = {
        rule.id: (rule.outcome.reason_template or "").strip()
        for rule in load_latest().rules
    }

    found: list[ReasonCode] = []
    for rule_id, code in sorted(STRUCTURAL_RULES.items(), key=lambda item: item[1]):
        statements = [templates.get(rule_id, "")]
        sources = [rule_id]
        alert = _ALERT_STATEMENTS.get(code)
        if alert:
            statements.append(alert)
            sources.append(f"group alert: {code.lower()}")
        found.append(
            ReasonCode(
                code=code,
                lanes=(Lane.GRAPH,),
                source=", ".join(sources),
                observation=_ALERT_OBSERVATIONS[code],
                statements=tuple(statements),
            )
        )
    return tuple(found)


#: Features the classifier may name, and the sentence it names them in.
#:
#: A statement earns a place here when a person holding the donation records
#: could verify it by counting, reading, or comparing against a reference the
#: sentence itself states. That is the whole test. A feature name with its
#: value after a colon fails it: it is a coordinate in the model's input space,
#: which is a property of the model rather than of the donation, and it cannot
#: be put to the person the alert is about.
#:
#: ``when_true`` wording carries no value at all. A boolean that came back false
#: is not a reason for anything, and printing "false" beside a label invites it
#: to be read as one.
_FEATURE_WORDING: dict[str, tuple[str, Render]] = {
    "amount": ("This donation is for {value}.", "rupiah"),
    "sender_type": ("The donor is recorded as {value}.", "plain"),
    "receiver_type": ("The recipient is recorded as {value}.", "plain"),
    "transaction_kind": ("The value moved as {value}.", "plain"),
    "channel": ("The record reached the system through the {value} channel.", "plain"),
    "is_round_amount": (
        "The amount is a round figure of a million rupiah or more.",
        "when_true",
    ),
    "amount_trailing_zeros": ("The amount ends in {value} zeros.", "plain"),
    "total_donasi_sender": (
        "Before this donation the donor had given {value} in total.",
        "rupiah",
    ),
    "jumlah_transaksi_sender": (
        "Before this donation the donor had made {value} donations.",
        "plain",
    ),
    "rata_rata_donasi_sender": (
        "The donor's earlier donations averaged {value}.",
        "rupiah",
    ),
    "std_donasi_sender": (
        "The donor's earlier donations varied by {value} either side of their "
        "own average donation.",
        "rupiah",
    ),
    "jumlah_donasi_30hari_sender": (
        "The donor made {value} donations in the 30 days before this one.",
        "plain",
    ),
    "selang_waktu_rata2_sender": (
        "The donor's earlier donations came {value} days apart on average.",
        "plain",
    ),
    "receiver_unik_per_sender": (
        "Before this donation the donor had given to {value} distinct "
        "recipients.",
        "plain",
    ),
    "max_donasi_satu_receiver": (
        "The most the donor had given to any single recipient is {value}.",
        "rupiah",
    ),
    "proporsi_donasi_terbesar_per_sender": (
        "The donor's largest single earlier donation is {value} of everything "
        "they had given.",
        "share",
    ),
    "sender_days_since_first_seen": (
        "The donor's first recorded donation was {value} days before this one.",
        "plain",
    ),
    "sender_is_first_donation": (
        "This is the donor's first recorded donation.",
        "when_true",
    ),
    "sender_unik_per_receiver": (
        "The recipient had received from {value} distinct donors.",
        "plain",
    ),
    "total_diterima_receiver": (
        "The recipient had received {value} in total.",
        "rupiah",
    ),
    "jumlah_transaksi_receiver": (
        "The recipient had received {value} donations.",
        "plain",
    ),
    "receiver_donor_concentration": (
        "The recipient's largest single donor accounts for {value} of its "
        "recorded funding.",
        "share",
    ),
    "receiver_new_donor_ratio_30d": (
        "{value} of the recipient's donors in the last 30 days were giving to "
        "it for the first time.",
        "share",
    ),
    "pair_prior_count": (
        "This donor had given to this recipient {value} times before.",
        "plain",
    ),
    "pair_prior_total": (
        "This donor had given this recipient {value} in total before.",
        "rupiah",
    ),
    "pair_is_first": (
        "This is the first donation between these two parties.",
        "when_true",
    ),
    "pair_days_since_last": (
        "The previous donation between these two parties was {value} days "
        "earlier.",
        "plain",
    ),
    "is_within_campaign_period": (
        "The donation falls inside a declared campaign period.",
        "when_true",
    ),
    "days_to_reporting_deadline": (
        "The donation is {value} days before the next reporting deadline.",
        "plain",
    ),
    "amount_to_limit_ratio": (
        "This donation is {value} of the statutory limit that applies to this "
        "donor and period.",
        "share",
    ),
    "cumulative_to_limit_ratio": (
        "Counting this donation, the donor has given {value} of the statutory "
        "limit that applies to them.",
        "share",
    ),
    "in_structuring_band": (
        "The amount sits between 90% and 100% of the applicable statutory "
        "limit.",
        "when_true",
    ),
    "extraction_confidence_min": (
        "The least confidently extracted field on this record was read with "
        "{value} confidence.",
        "share",
    ),
    "entity_resolution_confidence": (
        "The less confidently matched of the two parties resolved at {value}.",
        "share",
    ),
    "has_unresolved_entity": (
        "One of the two parties could not be matched to a known entity.",
        "when_true",
    ),
    "field_provenance_mix": (
        "{value} of the recorded fields were extracted from a document rather "
        "than submitted directly.",
        "share",
    ),
    "sender_out_degree": ("The donor has reached {value} distinct recipients.", "plain"),
    "receiver_in_degree": (
        "The recipient has drawn {value} distinct donors.",
        "plain",
    ),
    "receiver_in_degree_velocity_14d": (
        "{value} of the recipient's distinct donors first appeared in the last "
        "14 days.",
        "share",
    ),
    "pass_through_ratio": (
        "This donation is {value} of what the donor received in the week "
        "before it.",
        "share",
    ),
    "pass_through_lag_days": (
        "{value} days passed between the donor's most recent incoming payment "
        "and this donation.",
        "plain",
    ),
    "shared_counterparty_count": (
        "{value} other donors have also given to this recipient.",
        "plain",
    ),
}

#: The reference a figure has to be read against, where stating the number
#: alone would not be actionable. Only reference points that are definitional
#: appear here. A claim about what is *usual* would be a measurement, and this
#: system has not taken it.
_FEATURE_COMPARISON: dict[str, str] = {
    "amount_to_limit_ratio": (
        "1.0 is the limit itself. Which limit applies depends on the donor "
        "type and the period, and is stated by the rule that tests it."
    ),
    "cumulative_to_limit_ratio": (
        "1.0 is the limit itself, reached across every donation counted "
        "towards it rather than by this one alone."
    ),
    "receiver_donor_concentration": (
        "1.0 means a single donor supplied everything the recipient has "
        "recorded."
    ),
}

#: Features with no form a reader could check, and why. Catalogued so the
#: enumeration stays complete, and barred from reaching a reason so that the
#: enumeration does not amount to a list of things the system is willing to
#: say.
_NOT_STATEABLE: dict[str, str] = {
    "amount_log": (
        "a log transform of the amount, which restates a figure the record "
        "already carries in a form nobody reads"
    ),
    "sender_velocity_ratio": (
        "a rate measured against another rate; checking it means rebuilding "
        "both windows from a definition that lives only in the feature code"
    ),
    "receiver_amount_cv_30d": (
        "a coefficient of variation, which no reader can recompute from the "
        "records in front of them"
    ),
    "campaign_period_phase": (
        "a position within the campaign period expressed from 0 to 1; the "
        "date it is derived from is the checkable fact, and this is not it"
    ),
    "day_of_week": (
        "a calendar coordinate that says nothing on its own and, said aloud "
        "about a donation, reads as an insinuation the system cannot support"
    ),
    "hour_of_day": (
        "a calendar coordinate that says nothing on its own; an unusual hour "
        "is a claim that needs a threshold and a rule, not a bare number"
    ),
    "local_cluster_size": (
        "a graph quantity whose boundary is set by the traversal that computed "
        "it, so two readers counting by hand would not agree"
    ),
}


def _classifier_codes() -> tuple[ReasonCode, ...]:
    """One code per feature the classifier can name as an input.

    Enumerated against the feature set for the same reason the graph wording is
    derived from the rule set: a feature added to the model is a new sentence
    put in front of an analyst, and it arrives unreviewed rather than inheriting
    the acceptance given to the sentences already there.

    The wording, unlike the enumeration, is written by hand. There is no way to
    generate a sentence a person can check out of an identifier, and the
    attempt produces a variable name with a colon after it — which is the model
    internal this system says is not a reason.
    """
    from cakradana.features.definitions import catalogue as feature_catalogue
    from cakradana.features.definitions import feature_names

    names = feature_names()
    undecided = sorted(set(names) - set(_FEATURE_WORDING) - set(_NOT_STATEABLE))
    if undecided:
        raise ValueError(
            f"no wording decision has been made for {', '.join(undecided)}; a "
            f"feature the classifier can name has to either carry a sentence a "
            f"reader can check or be marked as carrying none"
        )

    specs = feature_catalogue()
    found: list[ReasonCode] = []
    for name in names:
        if name in _NOT_STATEABLE:
            found.append(
                ReasonCode(
                    code=name.upper(),
                    lanes=(Lane.CLASSIFIER,),
                    source=f"feature: {name}",
                    observation=_NOT_STATEABLE[name],
                    analyst_facing=False,
                )
            )
            continue
        statement, render = _FEATURE_WORDING[name]
        found.append(
            ReasonCode(
                code=name.upper(),
                lanes=(Lane.CLASSIFIER,),
                source=f"feature: {name}",
                observation=specs[name].description,
                statements=(statement,),
                comparison=_FEATURE_COMPARISON.get(name),
                render=render,
            )
        )
    return tuple(found)


@lru_cache(maxsize=1)
def catalogue() -> tuple[ReasonCode, ...]:
    """Every reason code the system can emit, ordered by code."""
    found = _FIXED + _graph_codes() + _classifier_codes()
    by_code: dict[str, ReasonCode] = {}
    for entry in found:
        if entry.code in by_code:
            raise ValueError(
                f"reason code {entry.code!r} is declared twice; a code with two "
                f"catalogue entries has two review states and neither governs"
            )
        by_code[entry.code] = entry
    return tuple(sorted(by_code.values(), key=lambda entry: entry.code))


def codes() -> tuple[str, ...]:
    return tuple(entry.code for entry in catalogue())


def stateable_codes() -> tuple[str, ...]:
    """The codes that can reach a person, and so are the ones under review.

    A code barred from being stated carries no wording, and there is nothing an
    analyst could accept or reject about it. Counting it towards review
    coverage would let the figure be raised by decisions about sentences
    nobody will ever read.
    """
    return tuple(entry.code for entry in catalogue() if entry.analyst_facing)


def entry_for(code: str) -> ReasonCode | None:
    return next((entry for entry in catalogue() if entry.code == code), None)


# ---------------------------------------------------------------------------
# Wording rules
# ---------------------------------------------------------------------------

#: Language that states what an observation means rather than what it is. A
#: reason carrying any of it has decided the case before the analyst read it,
#: and is a claim the system cannot support against a subject who contests it.
VERDICT_WORDS: tuple[str, ...] = (
    "suspicious",
    "suspected",
    "fraud",
    "fraudulent",
    "illegal",
    "unlawful",
    "violation",
    "offence",
    "offense",
    "guilty",
    "corrupt",
    "risky",
    "risk",
    "smurfing",
    "laundering",
    "kickback",
    "bribe",
    "proves",
    "confirms",
    "indicates",
    "therefore",
    "should be",
    "must be",
    "evidence of",
)

#: Score and band vocabulary. A reason explains a score; restating the score
#: inside it explains nothing and invites the reason to be read as the finding.
SCORE_WORDS: tuple[str, ...] = (
    "score",
    "scored",
    "probability",
    "percentile",
    "risk band",
    "score band",
    "flagged",
    "alert level",
)

#: Model internals. Properties of the model, not of the donation in front of
#: the reader, and not checkable by anybody the alert concerns.
INTERNAL_WORDS: tuple[str, ...] = (
    "feature importance",
    "importance",
    "coefficient",
    "shap",
    "gradient",
    "logit",
    "embedding",
    "eigen",
    "hyperparameter",
    "estimator",
    "n_estimators",
    "boosting",
    "z-score",
    "p-value",
)


def _hits(text: str, words: tuple[str, ...]) -> tuple[str, ...]:
    lowered = text.casefold()
    return tuple(
        word
        for word in words
        if re.search(rf"(?<![a-z]){re.escape(word)}(?![a-z])", lowered)
    )


#: A statement that is a label, a colon, and a value. The shape a generated
#: wording takes when nobody wrote one: `amount to limit ratio: 0.42.` names a
#: column of the model's input, not anything about the donation, and a reader
#: cannot check a quantity whose definition the sentence never states.
_LABEL_AND_VALUE = re.compile(r"[\w' ]{1,80}:\s*\{[^{}]*\}\.?\s*$")


def wording_defects(entry: ReasonCode) -> tuple[str, ...]:
    """What is wrong with a code's wording, stated so it can be acted on.

    Returns an empty tuple when the wording is well formed. Well formed is not
    the same as reviewed: these are the defects a machine can find, and they are
    the floor an analyst's reading starts from rather than a substitute for it.

    A code marked as never shown is checked for the opposite property. It must
    carry no wording, and the model refuses it at construction if it does; what
    is checked here is that it still says which quantity it stands for, because
    a barred code nobody can identify cannot be reconsidered later.
    """
    defects: list[str] = []

    if not entry.lanes:
        defects.append("names no lane, so the reader cannot tell what produced it")
    if not entry.source.strip():
        defects.append(
            "names no rule, alert, detector, or feature, so a disputed "
            "statement cannot be traced back to what produced it"
        )
    if not entry.observation.strip():
        defects.append("states no observation")

    if not entry.analyst_facing:
        return tuple(defects)

    if not entry.statements:
        defects.append("carries no wording at all")

    for statement in entry.statements:
        if not statement.strip():
            defects.append("has an empty wording template")
            continue
        if _LABEL_AND_VALUE.fullmatch(statement.strip()):
            defects.append(
                "reads as a label and a value rather than a statement about "
                "the donation, so it names a model input instead of an "
                "observation a person can check"
            )
        for label, words in (
            ("states a conclusion rather than an observation", VERDICT_WORDS),
            ("restates the score instead of explaining it", SCORE_WORDS),
            ("exposes a model internal rather than something observed", INTERNAL_WORDS),
        ):
            found = _hits(statement, words)
            if found:
                defects.append(f"{label}: {', '.join(found)}")

    return tuple(defects)
