"""The external reputation lane.

Reads adverse coverage about donors and turns it into a weak ranking signal.

This is the most dangerous lane in the system and the only one whose default is
not to run. The others reason about transactions the system has records of.
This one reasons about what has been *written* about a person, which means it
can be wrong in a way the others cannot: a donor who was reported on and
cleared, a donor who shares a name with someone else, or a donor targeted by a
campaign of hostile coverage all look the same to it.

Three constraints follow, and all three are enforced here rather than trusted
to whoever configures it.

It never produces a legal finding. The statute that would be relevant requires
a conviction with final legal force; reporting on an investigation, a named
suspect, or a pending prosecution does not meet that standard, and treating
coverage as though it did would be both wrong in law and defamatory.

It contributes the smallest share of any lane, because it rests on the weakest
evidence in the system.

It does not operate unless a set of conditions is explicitly met. The gate is
code rather than documentation, so switching it on is a deliberate act with a
record, not a config value someone flips while looking at something else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Iterable, Sequence

from cakradana.features import FeatureVector
from cakradana.rules.context import RuleContext
from cakradana.rules.engine import RuleEvaluation
from cakradana.scoring.composition import contribution_from, unavailable
from cakradana.scoring.result import Lane, LaneResult, Reason

#: Match confidence below which a coverage item is not attributed to a donor at
#: all. Name collision is the failure that turns this lane into a machine for
#: defaming people who share a name with someone in the news.
MIN_MATCH_CONFIDENCE = 0.95

#: Coverage older than this contributes nothing. A story from a decade ago says
#: little about a donation made last week, and letting it persist indefinitely
#: means a person can never stop being the subject of it.
MAX_AGE_DAYS = 365 * 3

#: Independent sources required before coverage counts at all. A single outlet
#: repeating one claim is one claim, however many times it is republished.
MIN_INDEPENDENT_SOURCES = 2


@dataclass(frozen=True)
class OperatingConditions:
    """What must be true before this lane runs.

    Every field defaults to false. The lane refuses to operate until each is
    explicitly set, and the reason it gives names the ones that are missing —
    so the state of the gate is legible from its output rather than only from
    whoever last edited the configuration.
    """

    #: A lawyer has assessed the exposure created by acting on adverse coverage.
    defamation_review_completed: bool = False
    #: Sources are named, and their selection can be explained to a subject who
    #: asks why these outlets and not others.
    source_list_published: bool = False
    #: Entity matching has been measured against a labelled sample, so the
    #: false-attribution rate is a number rather than a hope.
    matching_accuracy_measured: bool = False
    #: A subject can see what coverage was attributed to them and contest it.
    subject_access_route_exists: bool = False
    #: Coverage that is retracted or superseded is removed, and removal
    #: propagates to scores already computed from it.
    retraction_handling_implemented: bool = False
    #: Somebody is accountable for the lane's output by name.
    named_owner: str | None = None
    #: The lane's incremental value has been measured. If it adds nothing over
    #: the other lanes, the exposure buys nothing.
    lift_measured: bool = False

    def unmet(self) -> tuple[str, ...]:
        missing = []
        if not self.defamation_review_completed:
            missing.append("a defamation exposure review has not been completed")
        if not self.source_list_published:
            missing.append("the list of sources consulted has not been published")
        if not self.matching_accuracy_measured:
            missing.append("entity matching accuracy has not been measured")
        if not self.subject_access_route_exists:
            missing.append(
                "no route exists for a subject to see and contest what was "
                "attributed to them"
            )
        if not self.retraction_handling_implemented:
            missing.append("retracted coverage is not removed from existing scores")
        if not self.named_owner:
            missing.append("no named person is accountable for this lane's output")
        if not self.lift_measured:
            missing.append(
                "the lane's incremental value over the others has not been measured"
            )
        return tuple(missing)

    @property
    def satisfied(self) -> bool:
        return not self.unmet()


@dataclass(frozen=True)
class CoverageItem:
    """One piece of adverse coverage attributed to an entity."""

    entity_id: str
    source: str
    published_at: datetime
    headline: str
    url: str | None = None
    #: How confidently this item was attributed to this entity.
    match_confidence: float = 0.0
    #: Whether the item concerns an allegation, a charge, or an adjudicated
    #: outcome. Only the last is a fact about what someone did, and the lane
    #: says which it is rather than flattening them.
    stage: str = "allegation"
    retracted: bool = False


@dataclass
class CoverageIndex:
    """Coverage held about entities."""

    items: list[CoverageItem] = field(default_factory=list)

    def about(self, entity_id: str, *, as_of: datetime) -> list[CoverageItem]:
        """Coverage usable for an entity at a point in time.

        Filters on attribution confidence, retraction, age, and publication
        date. The last matters as much as the others: coverage published after
        a donation could not have been known when it was scored, and admitting
        it would make a historical score impossible to reproduce.
        """
        cutoff = as_of - timedelta(days=MAX_AGE_DAYS)
        return [
            item
            for item in self.items
            if item.entity_id == entity_id
            and not item.retracted
            and item.match_confidence >= MIN_MATCH_CONFIDENCE
            and cutoff <= item.published_at <= as_of
        ]

    def add(self, item: CoverageItem) -> None:
        self.items.append(item)

    def retract(self, url: str) -> int:
        """Mark coverage as retracted.

        Retraction is honoured by exclusion rather than deletion, so a score
        recorded earlier can still be explained by what was known then.
        """
        count = 0
        for index, item in enumerate(self.items):
            if item.url == url and not item.retracted:
                self.items[index] = CoverageItem(**{**item.__dict__, "retracted": True})
                count += 1
        return count


class ReputationLane:
    """Turns adverse coverage into a small, heavily-caveated ranking signal."""

    name = Lane.REPUTATION

    def __init__(
        self,
        coverage: CoverageIndex,
        conditions: OperatingConditions | None = None,
    ) -> None:
        self.coverage = coverage
        self.conditions = conditions or OperatingConditions()

    def evaluate(
        self,
        evaluation: RuleEvaluation,
        ctx: RuleContext,
        features: FeatureVector,
    ) -> LaneResult:
        unmet = self.conditions.unmet()
        if unmet:
            # Refused rather than silently degraded. The reason names what is
            # missing so the gate's state is visible in the output.
            return unavailable(
                Lane.REPUTATION,
                "not operating: " + "; ".join(unmet),
            )

        sender = ctx.donation.sender_ref
        if not sender.is_resolved:
            return unavailable(
                Lane.REPUTATION,
                "donor is unresolved, and coverage must not be attributed to a name",
            )

        items = self.coverage.about(sender.key, as_of=ctx.donation.occurred_at)
        if not items:
            return unavailable(Lane.REPUTATION, "no coverage matched this donor")

        sources = {item.source for item in items}
        if len(sources) < MIN_INDEPENDENT_SOURCES:
            # One outlet repeating a claim is one claim, however many times it
            # is republished.
            return unavailable(
                Lane.REPUTATION,
                f"coverage comes from {len(sources)} source; at least "
                f"{MIN_INDEPENDENT_SOURCES} independent sources are required",
            )

        intensity = self._intensity(items, sources)
        return contribution_from(
            Lane.REPUTATION, intensity, (self._reason(items, sources, ctx),)
        )

    def _intensity(
        self, items: Sequence[CoverageItem], sources: Iterable[str]
    ) -> float:
        """How strongly coverage weighs, saturating quickly.

        Adjudicated outcomes count for more than allegations, but even the
        strongest coverage cannot fill this lane's share: it is reporting about
        a person, and reporting is not a transaction the system observed.
        """
        weights = {"adjudicated": 1.0, "charged": 0.6, "allegation": 0.3}
        strongest = max(weights.get(item.stage, 0.3) for item in items)
        breadth = min(len(set(sources)) / 4.0, 1.0)
        return min(strongest * (0.6 + 0.4 * breadth), 1.0)

    def _reason(
        self,
        items: Sequence[CoverageItem],
        sources: Iterable[str],
        ctx: RuleContext,
    ) -> Reason:
        stages = {item.stage for item in items}
        # The wording states what the coverage is, not what the donor did.
        # "Reported as under investigation" and "has been investigated" are
        # different claims, and only the first is one this lane can support.
        stage_phrase = (
            "an adjudicated outcome"
            if "adjudicated" in stages
            else "charges" if "charged" in stages else "allegations"
        )
        return Reason(
            code="ADVERSE_COVERAGE",
            lane=Lane.REPUTATION,
            weight=0.2,
            statement=(
                f"{len(items)} item(s) of published coverage across "
                f"{len(set(sources))} independent sources report {stage_phrase} "
                f"concerning this donor. This is what has been written, not a "
                f"finding about what the donor did."
            ),
            comparison="Most donors attract no adverse coverage at all.",
            evidence_ref=f"donation:{ctx.donation.donation_id}",
        )
