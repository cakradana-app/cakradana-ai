"""The scoring service and its HTTP contract."""

from __future__ import annotations

from datetime import date

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from cakradana.calendar import CampaignPeriod, ElectoralCalendar  # noqa: E402
from cakradana.serving.api import TOKEN_ENV, create_app  # noqa: E402
from cakradana.serving.service import ScoringService  # noqa: E402

CONTEXT = "pemilu-2029"
TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
INDIVIDUAL_PARTY_LIMIT = 200_000_000


@pytest.fixture(autouse=True)
def service_token(monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, TOKEN)


@pytest.fixture
def calendar():
    return ElectoralCalendar(
        [
            CampaignPeriod(
                electoral_context=CONTEXT,
                start=date(2026, 9, 1),
                end=date(2026, 11, 24),
            )
        ]
    )


@pytest.fixture
def service(calendar):
    return ScoringService(calendar=calendar, require_verified_citations=False)


@pytest.fixture
def client(service):
    return TestClient(create_app(service))


def donation(
    donation_id: str = "d1",
    *,
    amount: int = 10_000_000,
    month: int = 3,
    sender: str = "donor-1",
) -> dict:
    stamp = f"2026-{month:02d}-01T00:00:00+07:00"
    return {
        "donation_id": donation_id,
        "sender_ref": {"entity_id": sender, "entity_type": "individual"},
        "receiver_ref": {"entity_id": "party-1", "entity_type": "political-party"},
        "amount_idr": amount,
        "occurred_at": stamp,
        "recorded_at": stamp,
        "channel": "digital-form",
        "electoral_context": CONTEXT,
    }


class TestOperationalEndpoints:
    def test_liveness_and_readiness_are_separate(self, client):
        """A running process with no rules would report every donation as
        carrying no findings, which reads exactly like a clean result."""
        assert client.get("/health").json() == {"status": "ok"}
        ready = client.get("/ready").json()
        assert ready["ready"] is True
        assert ready["rule_set"]
        assert ready["features"].startswith("features-")

    def test_readiness_needs_no_token(self, client):
        assert client.get("/ready").status_code == 200


class TestAuthentication:
    def test_an_unauthenticated_call_is_refused(self, client):
        response = client.post(
            "/v1/score", json={"request_id": "r", "donation": donation()}
        )
        assert response.status_code == 401

    def test_a_wrong_token_is_refused(self, client):
        response = client.post(
            "/v1/score",
            headers={"Authorization": "Bearer wrong"},
            json={"request_id": "r", "donation": donation()},
        )
        assert response.status_code == 401

    def test_an_unconfigured_token_refuses_everyone(self, client, monkeypatch):
        """An unset secret is a deployment mistake, and the safe reading is
        that nobody is authorised rather than that everybody is."""
        monkeypatch.delenv(TOKEN_ENV, raising=False)
        response = client.post(
            "/v1/score", headers=AUTH, json={"request_id": "r", "donation": donation()}
        )
        assert response.status_code == 503


class TestContract:
    def test_the_caller_sends_a_record_not_features(self, client):
        """The previous service demanded fourteen engineered inputs that no
        caller could produce, so the two services could not be connected."""
        payload = {**donation(), "total_donasi_sender": 5_000_000}
        response = client.post(
            "/v1/score", headers=AUTH, json={"request_id": "r", "donation": payload}
        )
        assert response.status_code == 422

    def test_every_response_carries_its_versions(self, client):
        body = client.post(
            "/v1/score", headers=AUTH, json={"request_id": "r", "donation": donation()}
        ).json()["result"]
        assert body["versions"]["rule_set"]
        assert body["versions"]["features"].startswith("features-")

    def test_naive_timestamps_are_refused(self, client):
        payload = {**donation(), "occurred_at": "2026-03-01T00:00:00"}
        response = client.post(
            "/v1/score", headers=AUTH, json={"request_id": "r", "donation": payload}
        )
        assert response.status_code == 422

    def test_a_donation_recorded_before_it_happened_is_refused(self, client):
        payload = {**donation(), "recorded_at": "2026-01-01T00:00:00+07:00"}
        response = client.post(
            "/v1/score", headers=AUTH, json={"request_id": "r", "donation": payload}
        )
        assert response.status_code == 422


class TestScoring:
    def test_cumulative_donations_are_caught_across_calls(self, client):
        """Each payment is lawful alone; the running total is not. This is the
        breach that no single-transaction check can see."""
        first = client.post(
            "/v1/score",
            headers=AUTH,
            json={"request_id": "r1", "donation": donation("d1", amount=150_000_000)},
        ).json()["result"]
        assert not first["legal_findings"]

        second = client.post(
            "/v1/score",
            headers=AUTH,
            json={
                "request_id": "r2",
                "donation": donation("d2", amount=120_000_000, month=5),
            },
        ).json()["result"]
        assert [f["rule_id"] for f in second["legal_findings"]] == ["RULE-T1-05"]

    def test_unevaluated_rules_are_always_surfaced(self, client):
        body = client.post(
            "/v1/score", headers=AUTH, json={"request_id": "r", "donation": donation()}
        ).json()["result"]
        assert body["indeterminate_rules"]

    def test_a_finding_names_its_statute_and_article(self, client):
        client.post(
            "/v1/score",
            headers=AUTH,
            json={"request_id": "r1", "donation": donation("d1", amount=150_000_000)},
        )
        body = client.post(
            "/v1/score",
            headers=AUTH,
            json={
                "request_id": "r2",
                "donation": donation("d2", amount=120_000_000, month=5),
            },
        ).json()["result"]
        finding = body["legal_findings"][0]
        assert finding["statute"] and finding["article"]
        assert finding["explanation"]


class TestBatch:
    def test_one_bad_item_does_not_fail_the_batch(self, service, client):
        """One unusable record must not discard everything submitted with it."""
        good = donation("d1")
        response = client.post(
            "/v1/score/batch",
            headers=AUTH,
            json={"request_id": "b", "donations": [good, donation("d2")]},
        )
        assert response.status_code == 200
        assert response.json()["succeeded"] == 2

    def test_a_batch_is_bounded(self, client):
        oversized = [donation(f"d{i}") for i in range(600)]
        response = client.post(
            "/v1/score/batch",
            headers=AUTH,
            json={"request_id": "b", "donations": oversized},
        )
        assert response.status_code == 422


class TestRescoring:
    def test_rescoring_keeps_the_previous_result(self, client):
        """An analyst who cleared an alert has to be able to see what they saw,
        and a subject contesting a score has to see the score they contest."""
        client.post(
            "/v1/score", headers=AUTH, json={"request_id": "r", "donation": donation()}
        )
        body = client.post(
            "/v1/rescore",
            headers=AUTH,
            json={
                "request_id": "r2",
                "donation": donation(),
                "reason": "late_arriving_data",
            },
        ).json()
        assert body["previous"] is not None
        assert body["result"]["donation_id"] == "d1"

    def test_an_unrecognised_reason_is_refused(self, client):
        response = client.post(
            "/v1/rescore",
            headers=AUTH,
            json={"request_id": "r", "donation": donation(), "reason": "because"},
        )
        assert response.status_code == 422


class TestIntrospection:
    def test_the_rule_set_is_inspectable(self, client):
        body = client.get("/v1/rules", headers=AUTH).json()
        assert body["version"]
        inactive = [r for r in body["rules"] if not r["active"]]
        assert inactive and all(r["inactive_reason"] for r in inactive)

    def test_unverified_citations_are_visible(self, client):
        """Whether a statutory citation has been reviewed is something an
        operator has to be able to see."""
        body = client.get("/v1/rules", headers=AUTH).json()
        tier1 = [r for r in body["rules"] if r["tier"] == 1]
        assert tier1 and any(not r["citation_verified"] for r in tier1)

    def test_model_info_reports_which_lanes_run(self, client):
        body = client.get("/v1/model-info", headers=AUTH).json()
        assert body["model_version"] is None
        assert body["lanes_available"] == ["graph"]

    def test_explain_returns_the_scoring_history(self, client):
        client.post(
            "/v1/score", headers=AUTH, json={"request_id": "r", "donation": donation()}
        )
        body = client.get("/v1/explain/d1", headers=AUTH).json()
        assert len(body["events"]) == 1

    def test_explaining_an_unknown_donation_is_a_not_found(self, client):
        assert client.get("/v1/explain/nope", headers=AUTH).status_code == 404
