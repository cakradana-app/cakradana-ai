"""Shared controlled vocabularies.

These enums are the single vocabulary used by extraction, storage, the rule
engine, and feature computation. Extraction prompts are generated from
``EntityType`` rather than restating a list, because a prompt that offers a
value the store rejects produces records that fail validation after the
expensive step has already run.

``UNKNOWN`` is a first-class value throughout. Nothing is silently coerced into
a known category to make a record fit.
"""

from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    """String-valued enum that serialises as its value."""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return str(self.value)


class EntityType(StrEnum):
    """Nature of a donating or receiving party.

    ``STATE_ENTERPRISE`` and ``VILLAGE_GOVERNMENT`` are separate from
    ``GOVERNMENT`` because the statutory prohibition names them separately, and
    a finding must be able to cite which prohibition applies.
    """

    INDIVIDUAL = "individual"
    CORPORATION = "corporation"
    ORGANIZATION = "organization"
    POLITICAL_PARTY = "political-party"
    GOVERNMENT = "government"
    STATE_ENTERPRISE = "state-enterprise"
    VILLAGE_GOVERNMENT = "village-government"
    FOREIGN_ENTITY = "foreign-entity"
    UNKNOWN = "unknown"


#: Types that count as a non-individual donor for limit selection. Grouped here
#: rather than inline in each rule so that adding a type cannot silently change
#: which limit applies without this list being reviewed.
NON_INDIVIDUAL_DONOR_TYPES = frozenset(
    {
        EntityType.CORPORATION,
        EntityType.ORGANIZATION,
        EntityType.FOREIGN_ENTITY,
    }
)

#: Types whose donations are prohibited outright by source, independent of
#: amount. Membership here does not itself produce a finding: a finding
#: requires a register lookup, because the type is an assertion about an entity
#: that must be evidenced rather than inferred.
PROHIBITED_SOURCE_TYPES = frozenset(
    {
        EntityType.GOVERNMENT,
        EntityType.STATE_ENTERPRISE,
        EntityType.VILLAGE_GOVERNMENT,
    }
)


class TransactionKind(StrEnum):
    """How value moved. Cash donations carry different traceability."""

    CASH = "cash"
    TRANSFER = "transfer"
    IN_KIND = "in_kind"
    UNKNOWN = "unknown"


class Channel(StrEnum):
    """How the record entered the system."""

    DIGITAL_FORM = "digital-form"
    PAPER_FORM = "paper-form"
    WEB_SCRAPE = "web-scrape"
    IMPORT = "import"


class TemporalPrecision(StrEnum):
    """Stated precision of a timestamp.

    Scanned forms routinely yield a date with no time. Storing that as midnight
    manufactures temporal clustering, which is precisely the signal the
    behavioural clustering heuristics test for. Features that depend on
    sub-day resolution must refuse to compute when precision is ``DAY``.
    """

    DAY = "day"
    HOUR = "hour"
    MINUTE = "minute"
    SECOND = "second"

    @property
    def has_time_of_day(self) -> bool:
        return self is not TemporalPrecision.DAY


class Provenance(StrEnum):
    """How a field's current value arose.

    Recorded per field, because a single record routinely mixes an extracted
    amount, a submitted date, and a human-corrected name, and how much each can
    be trusted differs.
    """

    EXTRACTED = "extracted"
    SUBMITTED = "submitted"
    SCRAPED = "scraped"
    HUMAN_CORRECTED = "human-corrected"
    DERIVED = "derived"


class Regime(StrEnum):
    """Which statutory limit regime governs a donation.

    Selection depends on the recipient's nature and on where the donation's date
    falls relative to a declared campaign period. ``INDETERMINATE`` is returned
    when the campaign period is unknown; it never falls back to the more
    permissive regime.
    """

    PARTY_ANNUAL = "party_annual"
    CAMPAIGN = "campaign"
    INDETERMINATE = "indeterminate"


class RuleOutcome(StrEnum):
    """Result of evaluating one rule against one donation.

    ``INDETERMINATE`` is distinct from ``PASS`` and the distinction is
    load-bearing: several statutory rules depend on reference data that may be
    absent, and an unevaluated prohibition must never be reported as a clean
    result.
    """

    LEGAL_FINDING = "LEGAL_FINDING"
    BEHAVIOURAL_SIGNAL = "BEHAVIOURAL_SIGNAL"
    PASS = "PASS"
    INDETERMINATE = "INDETERMINATE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class LabelValue(StrEnum):
    RISKY = "risky"
    NOT_RISKY = "not_risky"
    INDETERMINATE = "indeterminate"


class LabelSource(StrEnum):
    """Origin of a label.

    Sources are never merged. A heuristic label and an adjudicated dispute
    outcome are different kinds of evidence, and collapsing them makes it
    impossible to weight them correctly or to evaluate against human judgement
    alone.
    """

    RULE_TIER2 = "rule_tier2"
    ANALYST_DISPOSITION = "analyst_disposition"
    RECIPIENT_CONFIRMATION = "recipient_confirmation"
    DISPUTE_OUTCOME = "dispute_outcome"
    SYNTHETIC = "synthetic"


#: Sources that constitute human judgement about risk. Evaluation uses these
#: exclusively: a model measured against heuristic labels is measured on how
#: well it memorised the heuristics.
#:
#: ``RECIPIENT_CONFIRMATION`` is deliberately absent. A recipient confirming a
#: donation establishes that the transaction occurred, which is a different
#: claim from the transaction being legitimate — a smurfed donation is
#: genuinely received, and its recipient confirms it truthfully. Admitting
#: confirmations as negative labels would teach the model that verified
#: smurfing is clean.
HUMAN_LABEL_SOURCES = frozenset(
    {
        LabelSource.ANALYST_DISPOSITION,
        LabelSource.DISPUTE_OUTCOME,
    }
)
