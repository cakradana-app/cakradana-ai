"""Admitting adverse coverage.

The lane this feeds refuses to operate without controls that are not software.
These tests cover the part that is: which sources may be read, what must be
turned away, and how many claims a story republished ten times amounts to.
"""

from __future__ import annotations

import pytest

from cakradana.lanes.sources import (
    DECAY_HALF_LIFE_DAYS,
    MIN_STAGE_WEIGHT,
    CoverageIngest,
    RefusedCoverage,
    Source,
    SourceAllowlist,
    decayed_weight,
    event_key,
)
from cakradana.lanes.reputation import CoverageItem
from tests.conftest import at

ALLOWLIST = SourceAllowlist(
    [
        Source(name="Kompas", hosts=("kompas.com",)),
        Source(name="Tempo", hosts=("tempo.co",)),
        Source(name="Detik", hosts=("detik.com",)),
    ]
)


def ingest() -> CoverageIngest:
    return CoverageIngest(allowlist=ALLOWLIST)


def admit(worker: CoverageIngest, **overrides):
    defaults = {
        "entity_id": "e-sender",
        "url": "https://kompas.com/a",
        "headline": "Penyidik memeriksa dugaan penyalahgunaan dana kampanye",
        "body": "Laporan lengkap mengenai pemeriksaan tersebut.",
        "published_at": at(2026, 5, 1),
        "stage": "charged",
        "match_confidence": 0.99,
    }
    return worker.admit(**{**defaults, **overrides})


class TestAllowlist:
    def test_only_configured_hosts_are_read(self):
        """Arbitrary search over a person's name finds the worst thing anyone
        has written about anyone who shares it."""
        with pytest.raises(RefusedCoverage, match="not a configured source"):
            admit(ingest(), url="https://example.org/rumours")

    def test_a_lookalike_host_is_not_the_source(self):
        # A suffix match would admit kompas.com.evil.example.
        assert not ALLOWLIST.permits("https://kompas.com.evil.example/a")
        assert ALLOWLIST.permits("https://kompas.com/a")
        assert ALLOWLIST.permits("https://www.kompas.com/a")

    def test_the_list_can_be_published(self):
        """One operating condition is that a subject can be told why these
        outlets and not others, which is answerable only if the system can
        state the list."""
        assert ALLOWLIST.published() == ("Detik", "Kompas", "Tempo")


class TestSelfReference:
    def test_coverage_derived_from_this_system_is_refused(self):
        """A story written from a flag this system raised, read back as
        evidence, is the system citing itself — while looking exactly like
        independent corroboration."""
        with pytest.raises(RefusedCoverage, match="citing itself"):
            admit(
                ingest(),
                body="Menurut sistem Cakradana, donasi tersebut ditandai berisiko.",
            )

    def test_a_headline_referring_to_it_is_refused_too(self):
        with pytest.raises(RefusedCoverage, match="citing itself"):
            admit(ingest(), headline="Cakradana flags donation to party")


class TestLegalStanding:
    def test_a_bare_mention_carries_no_weight(self):
        """Somebody's name appearing near a subject is not evidence about the
        subject."""
        with pytest.raises(RefusedCoverage, match="no evidential weight"):
            admit(ingest(), stage="mentioned")

    def test_the_floor_excludes_only_the_weakest_stage(self):
        assert MIN_STAGE_WEIGHT <= 0.3

    def test_a_conviction_outweighs_an_allegation(self):
        adjudicated = CoverageItem(
            entity_id="e",
            source="Kompas",
            published_at=at(2026, 5, 1),
            headline="h",
            stage="adjudicated",
        )
        allegation = CoverageItem(
            entity_id="e",
            source="Kompas",
            published_at=at(2026, 5, 1),
            headline="h",
            stage="allegation",
        )
        as_of = at(2026, 6, 1)
        assert decayed_weight(adjudicated, as_of=as_of) > decayed_weight(
            allegation, as_of=as_of
        )


class TestDecay:
    def test_coverage_halves_over_the_half_life(self):
        item = CoverageItem(
            entity_id="e",
            source="Kompas",
            published_at=at(2024, 6, 1),
            headline="h",
            stage="adjudicated",
        )
        fresh = CoverageItem(
            entity_id="e",
            source="Kompas",
            published_at=at(2026, 6, 1),
            headline="h",
            stage="adjudicated",
        )
        as_of = at(2026, 6, 1)
        assert decayed_weight(item, as_of=as_of) == pytest.approx(
            decayed_weight(fresh, as_of=as_of) * 0.5, rel=0.02
        )

    def test_the_half_life_is_measured_in_years_not_decades(self):
        assert DECAY_HALF_LIFE_DAYS <= 365 * 3


class TestDeduplication:
    def test_one_source_reporting_an_event_twice_is_one_claim(self):
        """Otherwise the lane measures how much attention a story got, and
        attention is what a coordinated campaign manufactures."""
        worker = ingest()
        admit(worker)
        with pytest.raises(RefusedCoverage, match="not a second claim"):
            admit(worker, url="https://kompas.com/b")

    def test_a_second_outlet_reporting_the_same_event_is_a_second_source(self):
        worker = ingest()
        admit(worker)
        item = admit(worker, url="https://tempo.co/a")
        assert item.source == "Tempo"
        key = event_key(
            "Penyidik memeriksa dugaan penyalahgunaan dana kampanye",
            "e-sender",
            at(2026, 5, 1),
        )
        assert worker.independent_sources_for(key) == 2

    def test_a_rewritten_headline_is_still_the_same_event(self):
        first = event_key("Penyidik memeriksa dugaan korupsi dana", "e", at(2026, 5, 1))
        reworded = event_key("Dugaan korupsi dana, penyidik memeriksa", "e", at(2026, 5, 2))
        assert first == reworded

    def test_the_same_story_a_year_later_is_a_development(self):
        first = event_key("Penyidik memeriksa dugaan korupsi dana", "e", at(2026, 5, 1))
        later = event_key("Penyidik memeriksa dugaan korupsi dana", "e", at(2027, 5, 1))
        assert first != later

    def test_a_different_subject_is_a_different_event(self):
        mine = event_key("Penyidik memeriksa dugaan korupsi dana", "e-1", at(2026, 5, 1))
        theirs = event_key("Penyidik memeriksa dugaan korupsi dana", "e-2", at(2026, 5, 1))
        assert mine != theirs
