"""Where adverse coverage may come from, and how it is admitted.

The lane that consumes this refuses to run without a defamation review, a
published source list, measured matching accuracy, a subject access route,
retraction handling, a named owner, and measured lift. None of those are
software, so this module cannot make the lane operate. What it does is make the
coverage side of it complete and inspectable, so that switching it on later is a
decision about controls rather than a decision to write code under pressure.

Three constraints hold regardless of whether the lane ever runs.

Only configured sources are consulted. Arbitrary web search over a person's name
is not a research method, it is a machine for finding the worst thing anyone has
ever written about someone who shares their name.

Coverage derived from Cakradana's own output is refused. A story written from a
flag this system raised, read back in as evidence, is the system citing itself —
and it would look exactly like independent corroboration.

Ten articles about one investigation are one signal. Without deduplication by
underlying event, the lane measures how much attention a story got rather than
anything about conduct, and attention is precisely what a coordinated campaign
can manufacture.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Iterable, Sequence
from urllib.parse import urlparse

from cakradana.lanes.reputation import CoverageIndex, CoverageItem

#: Legal standing, and what each is worth. Reporting that somebody has been
#: named a suspect is not reporting that they did anything, and the gap between
#: the top and bottom of this table is the difference between a fact and a
#: rumour with a byline.
STAGE_WEIGHTS: dict[str, float] = {
    "adjudicated": 1.0,
    "charged": 0.6,
    "allegation": 0.3,
    "mentioned": 0.0,
}

#: Coverage below this contributes nothing at all. "Mentioned in reporting
#: without formal status" is somebody's name appearing near a subject, which is
#: not evidence of anything.
MIN_STAGE_WEIGHT = 0.3

#: Half-life for decay. An investigation reported six years ago says less about
#: a donation made last week than one reported six months ago.
DECAY_HALF_LIFE_DAYS = 365 * 2


@dataclass(frozen=True)
class Source:
    """A publication this system is configured to read."""

    name: str
    #: Hosts this source publishes on. Matched exactly rather than by suffix: a
    #: suffix match on "kompas.com" also admits "kompas.com.evil.example".
    hosts: tuple[str, ...]
    #: Whether the source is editorially independent of this system's operator.
    #: Recorded rather than assumed, because an operator's own newsletter
    #: reporting its own findings is not a second source.
    independent: bool = True
    note: str | None = None

    def covers(self, url: str) -> bool:
        host = (urlparse(url).hostname or "").lower()
        return host in self.hosts or any(host == f"www.{h}" for h in self.hosts)


class SourceAllowlist:
    """The sources that may be consulted, and nothing else."""

    def __init__(self, sources: Iterable[Source] = ()) -> None:
        self.sources: tuple[Source, ...] = tuple(sources)

    def for_url(self, url: str) -> Source | None:
        for source in self.sources:
            if source.covers(url):
                return source
        return None

    def permits(self, url: str) -> bool:
        return self.for_url(url) is not None

    def published(self) -> tuple[str, ...]:
        """The list, for publishing.

        One of the lane's operating conditions is that the source list has been
        published and its selection can be explained to a subject who asks why
        these outlets and not others. That is answerable only if the list is
        something the system can state.
        """
        return tuple(sorted(source.name for source in self.sources))


class RefusedCoverage(ValueError):
    """Raised when an item cannot be admitted, with the reason."""


#: Markers that a piece of coverage was written from this system's output.
#: Deliberately broad: a false refusal costs one article, and a false admission
#: is the system reading its own conclusion back as independent evidence.
SELF_REFERENCE = re.compile(
    r"cakradana|sistem\s+deteksi\s+risiko\s+donasi|risk\s+score\s+of\s+\d+",
    re.IGNORECASE,
)


def event_key(item_headline: str, entity_id: str, published_at: datetime) -> str:
    """An identity for the underlying event rather than the article.

    Ten outlets reporting one investigation on the same day share a key, so the
    lane counts the investigation once. Built from the significant words in the
    headline, so re-wordings collapse together while genuinely different events
    do not.
    """
    words = sorted(
        word
        for word in re.findall(r"\w+", item_headline.lower())
        if len(word) > 4
    )
    digest = hashlib.sha256()
    digest.update(entity_id.encode())
    digest.update(b"|")
    # Bucketed by week: the same event reported on consecutive days is one
    # event, and reported a year apart is a development worth counting again.
    digest.update(published_at.strftime("%G-W%V").encode())
    digest.update(b"|")
    digest.update("+".join(words[:8]).encode())
    return digest.hexdigest()[:16]


@dataclass
class CoverageIngest:
    """Admits coverage into the index, or refuses it with a reason."""

    allowlist: SourceAllowlist
    #: Event keys already admitted, so a story republished elsewhere adds a
    #: source without adding a claim.
    seen_events: dict[str, set[str]] = field(default_factory=dict)

    def admit(
        self,
        *,
        entity_id: str,
        url: str,
        headline: str,
        body: str,
        published_at: datetime,
        stage: str,
        match_confidence: float,
        retrieved_at: datetime | None = None,
    ) -> CoverageItem:
        source = self.allowlist.for_url(url)
        if source is None:
            raise RefusedCoverage(
                f"{url} is not a configured source; arbitrary search over a "
                f"person's name finds the worst thing written about anyone who "
                f"shares it"
            )

        if SELF_REFERENCE.search(body) or SELF_REFERENCE.search(headline):
            raise RefusedCoverage(
                "this coverage appears to derive from Cakradana's own output; "
                "reading it back would be the system citing itself while looking "
                "like independent corroboration"
            )

        if STAGE_WEIGHTS.get(stage, 0.0) < MIN_STAGE_WEIGHT:
            raise RefusedCoverage(
                f"stage '{stage}' carries no evidential weight; a name appearing "
                f"near a subject is not evidence about the subject"
            )

        key = event_key(headline, entity_id, published_at)
        sources_for_event = self.seen_events.setdefault(key, set())
        already = source.name in sources_for_event
        sources_for_event.add(source.name)
        if already:
            raise RefusedCoverage(
                "this source has already reported this event; republication is "
                "not a second claim"
            )

        return CoverageItem(
            entity_id=entity_id,
            source=source.name,
            published_at=published_at,
            headline=headline,
            url=url,
            match_confidence=match_confidence,
            stage=stage,
        )

    def independent_sources_for(self, event: str) -> int:
        return len(self.seen_events.get(event, ()))


def decayed_weight(item: CoverageItem, *, as_of: datetime) -> float:
    """What a piece of coverage is worth now.

    Severity by legal standing, halved every two years. Both halves matter: a
    conviction is not an allegation, and a six-year-old story is not a current
    one.
    """
    base = STAGE_WEIGHTS.get(item.stage, 0.0)
    age_days = max((as_of - item.published_at).total_seconds() / 86_400, 0.0)
    return base * 0.5 ** (age_days / DECAY_HALF_LIFE_DAYS)


def build_index(items: Sequence[CoverageItem]) -> CoverageIndex:
    index = CoverageIndex()
    for item in items:
        index.add(item)
    return index
