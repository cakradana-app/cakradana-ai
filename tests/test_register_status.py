"""Whether the reference lists are usable, and whether anyone can tell.

A register that quietly stops being refreshed produces no error. The dependent
rules return indeterminate, nothing raises, and the queue simply stops
containing that kind of finding — which is indistinguishable, from outside,
from a population in which nobody is a prohibited source.

The freshness rule itself was already enforced. What is tested here is that its
state is legible: that a stale register reports as unusable rather than merely
old, that an unsupplied one and an expired one are reported as the same kind of
problem because the rules treat them identically, and that each says which rules
stop producing findings as a result.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from cakradana.registers import (
    Register,
    RegisterEntry,
    RegisterSet,
    empty_register_set,
)

NOW = datetime(2026, 8, 17, tzinfo=timezone.utc)


def supplied(**kwargs) -> Register:
    defaults = {
        "entries": [
            RegisterEntry(entity_id="e-1", canonical_name="PT PLN (Persero)")
        ],
        "available": True,
        "refreshed_at": NOW - timedelta(days=5),
        "max_age": timedelta(days=30),
    }
    defaults.update(kwargs)
    return Register(RegisterSet.PROHIBITED_SOURCE, **defaults)


class TestOneRegister:
    def test_a_current_register_is_usable_and_says_nothing_further(self):
        status = supplied().status(NOW)
        assert status["usable"] is True
        assert status["stale"] is False
        assert status["reason"] is None
        assert status["consequence"] is None

    def test_a_stale_register_is_unusable_rather_than_merely_old(self):
        status = supplied(refreshed_at=NOW - timedelta(days=90)).status(NOW)
        assert status["stale"] is True
        assert status["usable"] is False
        assert "cannot be relied on" in status["reason"]

    def test_an_unsupplied_register_is_the_same_kind_of_problem(self):
        """The rules treat "never supplied" and "expired" identically, so a
        status that separated them without saying so would invite reading one
        as the other."""
        status = Register(RegisterSet.PROHIBITED_SOURCE).status(NOW)
        assert status["usable"] is False
        assert status["available"] is False
        assert status["stale"] is False
        assert status["reason"]

    def test_an_unusable_register_names_what_stops_working(self):
        status = Register(RegisterSet.FINAL_CONVICTIONS).status(NOW)
        assert "final_convictions" in status["consequence"]
        assert "indeterminate" in status["consequence"]

    def test_a_register_with_no_horizon_never_goes_stale(self):
        """Some lists genuinely do not expire. Reporting one as stale because
        nobody set a horizon would train operators to ignore the field."""
        status = supplied(max_age=None, refreshed_at=None).status(NOW)
        assert status["stale"] is False
        assert status["usable"] is True

    def test_a_register_with_a_horizon_and_no_refresh_date_is_stale(self):
        """Not knowing when a list was last updated is not evidence that it is
        current."""
        assert supplied(refreshed_at=None).status(NOW)["stale"] is True

    def test_the_status_reports_the_horizon_it_was_judged_against(self):
        status = supplied().status(NOW)
        assert status["max_age_days"] == 30
        assert status["refreshed_at"]

    def test_a_supplied_but_empty_register_is_still_usable(self):
        """An empty list is a statement — nobody is on it. "There is no list"
        is a different statement, and the two must not collapse."""
        status = supplied(entries=[]).status(NOW)
        assert status["usable"] is True
        assert status["entries"] == 0

    def test_a_non_authoritative_register_says_so(self):
        """A fixture lets the dependent rules run end to end, but a finding
        drawn from one is a demonstration, not enforcement."""
        status = supplied(authoritative=False).status(NOW)
        assert status["authoritative"] is False


class TestTheSet:
    def test_every_declared_register_appears_even_when_unsupplied(self):
        found = {status["name"] for status in empty_register_set().status(NOW)}
        assert found == {
            RegisterSet.PROHIBITED_SOURCE,
            RegisterSet.FINAL_CONVICTIONS,
        }

    def test_the_default_posture_is_that_nothing_is_usable(self):
        assert all(not s["usable"] for s in empty_register_set().status(NOW))

    def test_unusable_registers_come_first(self):
        """The list is read to find what is wrong, and a reader who has to
        scan past healthy rows will stop scanning."""
        statuses = RegisterSet([supplied()]).status(NOW)
        assert statuses[0]["name"] == RegisterSet.FINAL_CONVICTIONS
        assert statuses[0]["usable"] is False
        assert statuses[-1]["usable"] is True

    def test_a_supplied_register_replaces_the_declared_placeholder(self):
        statuses = {s["name"]: s for s in RegisterSet([supplied()]).status(NOW)}
        assert statuses[RegisterSet.PROHIBITED_SOURCE]["usable"] is True
        assert statuses[RegisterSet.PROHIBITED_SOURCE]["entries"] == 1


class TestTheEndpoint:
    @pytest.fixture
    def client(self, monkeypatch):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from cakradana.serving.api import TOKEN_ENV, create_app
        from cakradana.serving.service import ScoringService

        monkeypatch.setenv(TOKEN_ENV, "test-token")
        return TestClient(
            create_app(ScoringService(require_verified_citations=False))
        )

    def test_the_service_reports_what_it_is_evaluating_against(self, client):
        response = client.get(
            "/v1/registers", headers={"Authorization": "Bearer test-token"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 2
        assert body["usable"] == 0

    def test_the_report_names_which_findings_will_not_be_raised(self, client):
        body = client.get(
            "/v1/registers", headers={"Authorization": "Bearer test-token"}
        ).json()
        assert set(body["rules_affected"]) == {
            RegisterSet.PROHIBITED_SOURCE,
            RegisterSet.FINAL_CONVICTIONS,
        }

    def test_the_report_describes_the_registers_actually_in_use(self, monkeypatch):
        """Held in one place rather than two. A second copy would let an
        operator read a freshness report about registers the engine is not
        using."""
        pytest.importorskip("fastapi")
        from cakradana.serving.service import ScoringService

        service = ScoringService(
            registers=RegisterSet([supplied()]), require_verified_citations=False
        )
        assert service.registers is service.scorer.engine.registers
        usable = [s for s in service.registers.status(NOW) if s["usable"]]
        assert [s["name"] for s in usable] == [RegisterSet.PROHIBITED_SOURCE]

    def test_the_report_needs_a_token(self, client):
        assert client.get("/v1/registers").status_code == 401
