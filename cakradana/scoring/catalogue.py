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

from pydantic import BaseModel, ConfigDict

from cakradana.scoring.result import Lane


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
    #: above it.
    statements: tuple[str, ...]

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
            "this donation is one leg of a chain passing through intermediate "
            "parties"
        ),
        statements=(
            "This donation is one leg of a chain of {donations} passing "
            "through {counterparties} intermediate entities.",
        ),
    ),
    ReasonCode(
        code="UNUSUAL_COMBINATION",
        lanes=(Lane.ANOMALY,),
        source="anomaly detector",
        observation=(
            "the donation sits outside the range the anomaly lane treats as "
            "ordinary, matching no known pattern"
        ),
        statements=(
            "This donation's combination of amount, donor history, and "
            "recipient pattern is unlike the donations around it, without "
            "matching any known pattern.",
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
            "the classifier ranked the donation highly but named no input to "
            "attribute it to"
        ),
        statements=(
            "The model ranked this donation above most others on its "
            "combination of donor history, recipient pattern, and amount.",
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


def _classifier_codes() -> tuple[ReasonCode, ...]:
    """One code per feature the classifier can name as an input.

    Derived from the feature set for the same reason the graph wording is
    derived from the rule set: a feature added to the model is a new sentence
    put in front of an analyst, and it arrives unreviewed rather than inheriting
    the acceptance given to the sentences already there.
    """
    from cakradana.features.definitions import feature_names

    return tuple(
        ReasonCode(
            code=name.upper(),
            lanes=(Lane.CLASSIFIER,),
            source=f"feature: {name}",
            observation=f"the value of {name.replace('_', ' ')} for this donation",
            statements=(f"{name.replace('_', ' ')}: {{value}}.",),
        )
        for name in feature_names()
    )


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


def wording_defects(entry: ReasonCode) -> tuple[str, ...]:
    """What is wrong with a code's wording, stated so it can be acted on.

    Returns an empty tuple when the wording is well formed. Well formed is not
    the same as reviewed: these are the defects a machine can find, and they are
    the floor an analyst's reading starts from rather than a substitute for it.
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
    if not entry.statements:
        defects.append("carries no wording at all")

    for statement in entry.statements:
        if not statement.strip():
            defects.append("has an empty wording template")
            continue
        for label, words in (
            ("states a conclusion rather than an observation", VERDICT_WORDS),
            ("restates the score instead of explaining it", SCORE_WORDS),
            ("exposes a model internal rather than something observed", INTERNAL_WORDS),
        ):
            found = _hits(statement, words)
            if found:
                defects.append(f"{label}: {', '.join(found)}")

    return tuple(defects)
