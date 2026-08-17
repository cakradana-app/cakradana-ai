"""HTTP interface to the scoring service.

The contract is versioned in the path and changes only additively within a
version. A caller integrating against it should never have a field's meaning
change under them, because the consumer of these results makes decisions about
named people.

Service-to-service calls are authenticated. This endpoint returns statements
about identifiable individuals and their political giving, so an unauthenticated
one would be a disclosure channel regardless of what the network in front of it
is assumed to do.
"""

from __future__ import annotations

import os

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from cakradana.serving.schemas import (
    BatchItemResult,
    BatchScoreRequest,
    HealthResponse,
    ModelInfoResponse,
    ReadyResponse,
    RescoreRequest,
    ScoreRequest,
)
from cakradana.serving.service import ScoringService, ServiceNotReady

#: Environment variable holding the shared service token.
TOKEN_ENV = "CAKRADANA_SERVICE_TOKEN"

API_PREFIX = "/v1"

#: Starlette renamed this constant; both spellings mean 422, and referring to
#: the old one under the new library raises a deprecation warning on every
#: validation failure. The number is the contract either way.
UNPROCESSABLE = 422


def _path(item: dict) -> str:
    """A validation error's field, as the caller wrote it.

    ``body`` is dropped from the front: the caller knows it sent a body, and
    what it needs is which of its own keys was the problem.
    """
    parts = [str(part) for part in item.get("loc", ()) if part != "body"]
    return ".".join(parts) or "body"


def create_app(service: ScoringService | None = None) -> FastAPI:
    """Build the application.

    The service is injected so that tests exercise the same wiring the process
    uses, rather than a parallel arrangement that can drift from it.
    """
    app = FastAPI(
        title="Cakradana scoring service",
        version="1.0",
        summary=(
            "Evaluates donations against statutory rules and behavioural "
            "lanes. Outputs prioritise donations for human review; they do "
            "not determine that an offence occurred."
        ),
    )
    app.state.service = service or ScoringService()

    def current_service() -> ScoringService:
        return app.state.service

    def require_token(authorization: str = Header(default="")) -> None:
        # Declared as a default rather than inside an annotation: this module
        # uses postponed annotation evaluation, under which a dependency that
        # closes over local state cannot be resolved from the module namespace
        # and is silently demoted to a query parameter.
        """Reject unauthenticated callers.

        When no token is configured the service refuses every request rather
        than accepting all of them. An unset secret is a deployment mistake,
        and the safe reading of it is that nobody is authorised yet.
        """
        expected = os.environ.get(TOKEN_ENV)
        if not expected:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    f"{TOKEN_ENV} is not configured; the service will not "
                    f"accept calls until a service token is set"
                ),
            )
        if authorization != f"Bearer {expected}":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="a valid service token is required",
            )

    # -- error contract --------------------------------------------------

    @app.exception_handler(RequestValidationError)
    def invalid_request(_: Request, error: RequestValidationError) -> JSONResponse:
        """Separate "I do not understand this field" from "this value is wrong".

        A field the schema does not define means the caller is speaking a
        different version of this contract, and every value in the request is
        then suspect — including the ones that parsed. That is a 400. A missing
        or malformed value in a request whose shape is right is a 422, and the
        field is named so the caller does not have to guess which one.
        """
        unknown = [
            _path(item) for item in error.errors() if item["type"] == "extra_forbidden"
        ]
        if unknown:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "error": {
                        "code": "unknown_field",
                        "message": (
                            "the request contains fields this contract does not "
                            "define; it is refused rather than partly applied, "
                            "because a caller sending unknown fields may be "
                            "expecting behaviour this version does not have"
                        ),
                        "fields": unknown,
                    }
                },
            )
        return JSONResponse(
            status_code=UNPROCESSABLE,
            content={
                "error": {
                    "code": "invalid_request",
                    "message": "the request could not be accepted as submitted",
                    "fields": [_path(item) for item in error.errors()],
                    "detail": [
                        {"field": _path(item), "problem": item["msg"]}
                        for item in error.errors()
                    ],
                }
            },
        )

    @app.exception_handler(Exception)
    def unhandled(_: Request, error: Exception) -> JSONResponse:
        """Anything unforeseen is the server's fault, and says so.

        Reporting an internal failure as a 4xx tells the caller to change its
        request, which will not help, and hides the fault from anyone counting
        server errors. The exception text is not returned: it is the one place
        a stack detail or a donor's name could reach an unintended reader.
        """
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": "internal_error",
                    "message": (
                        "the request could not be completed; this is a fault in "
                        "the service and the request was not scored"
                    ),
                }
            },
        )

    # -- operational -----------------------------------------------------

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Liveness only: the process is running."""
        return HealthResponse(status="ok")

    @app.get("/ready", response_model=ReadyResponse)
    def ready(
        svc: ScoringService = Depends(current_service),
    ) -> ReadyResponse:
        """Readiness: the process can produce a meaningful answer.

        Kept separate from liveness because a running process with no rules
        loaded would report every donation as carrying no findings, which reads
        exactly like a clean result.
        """
        return ReadyResponse(
            ready=svc.is_ready,
            rule_set=svc.ruleset.version if svc.ruleset else None,
            features=svc.feature_set_version,
            model=svc.artifact.version if svc.artifact else None,
            detail=svc.readiness_detail(),
        )

    @app.get(f"{API_PREFIX}/model-info", response_model=ModelInfoResponse)
    def model_info(
        svc: ScoringService = Depends(current_service),
        _: None = Depends(require_token),
    ) -> ModelInfoResponse:
        return ModelInfoResponse(
            model_version=svc.artifact.version if svc.artifact else None,
            rule_set_version=svc.ruleset.version,
            feature_set_version=svc.feature_set_version,
            threshold=svc.threshold,
            shipped_on_merit=(
                svc.artifact.shipped_on_merit if svc.artifact else None
            ),
            lanes_available=svc.available_lanes,
        )

    @app.get(f"{API_PREFIX}/registers")
    def registers(
        svc: ScoringService = Depends(current_service),
        _: None = Depends(require_token),
    ) -> dict:
        """Which reference lists are usable, and which are not.

        A register that quietly went stale looks, from outside, exactly like a
        population in which nobody is a prohibited source: the dependent rules
        return indeterminate, nothing raises, and the queue simply stops
        containing that kind of finding. Nothing else in the system would show
        it, so it is reported here.

        Unusable registers come first, because this list is read to find what
        is wrong and a reader who has to scan past healthy rows will stop
        scanning.
        """
        found = svc.registers.status()
        return {
            "registers": list(found),
            "usable": sum(1 for item in found if item["usable"]),
            "total": len(found),
            "rules_affected": [
                item["name"] for item in found if not item["usable"]
            ],
        }

    @app.get(f"{API_PREFIX}/rules")
    def rules(
        svc: ScoringService = Depends(current_service),
        _: None = Depends(require_token),
    ) -> dict:
        """The active rule set, so a finding can be traced to its definition."""
        return {
            "version": svc.ruleset.version,
            "rules": [
                {
                    "id": rule.id,
                    "tier": rule.tier,
                    "title": rule.title,
                    "typology": rule.typology,
                    "active": rule.active,
                    "inactive_reason": rule.inactive_reason,
                    "citation_verified": rule.is_verified,
                    "effective_from": rule.effective.from_.isoformat(),
                    "effective_to": (
                        rule.effective.to.isoformat() if rule.effective.to else None
                    ),
                    "statute": (
                        rule.legal_basis.statute if rule.legal_basis else None
                    ),
                    "article": (
                        rule.legal_basis.article if rule.legal_basis else None
                    ),
                }
                for rule in svc.ruleset.rules
            ],
        }

    # -- scoring ---------------------------------------------------------

    @app.post(f"{API_PREFIX}/score")
    def score(
        request: ScoreRequest,
        svc: ScoringService = Depends(current_service),
        _: None = Depends(require_token),
    ) -> dict:
        try:
            result = svc.score(request.donation)
        except ServiceNotReady as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)
            ) from error
        return {"request_id": request.request_id, "result": result.model_dump(mode="json")}

    @app.post(f"{API_PREFIX}/score/batch")
    def score_batch(
        request: BatchScoreRequest,
        svc: ScoringService = Depends(current_service),
        _: None = Depends(require_token),
    ) -> dict:
        """Score a batch, reporting per item.

        Never all-or-nothing: one unusable record must not discard the results
        for everything submitted alongside it.
        """
        items = [
            BatchItemResult(
                donation_id=payload.donation_id,
                ok=result is not None,
                result=result.model_dump(mode="json") if result else None,
                error=error,
            )
            for payload, result, error in svc.score_many(request.donations)
        ]
        return {
            "request_id": request.request_id,
            "items": [item.model_dump(mode="json") for item in items],
            "succeeded": sum(1 for item in items if item.ok),
            "failed": sum(1 for item in items if not item.ok),
        }

    @app.post(f"{API_PREFIX}/rescore")
    def rescore(
        request: RescoreRequest,
        svc: ScoringService = Depends(current_service),
        _: None = Depends(require_token),
    ) -> dict:
        """Score again, keeping the previous result.

        A new scoring event is appended rather than replacing what came before.
        An analyst who cleared an alert has to be able to see what they saw,
        and a subject contesting a score has to be able to see the score they
        are contesting.
        """
        previous = svc.latest_for(request.donation.donation_id)
        try:
            result = svc.score(request.donation, remember=False)
        except ServiceNotReady as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)
            ) from error
        return {
            "request_id": request.request_id,
            "reason": request.reason,
            "note": request.note,
            "result": result.model_dump(mode="json"),
            "previous": previous.model_dump(mode="json") if previous else None,
        }

    @app.get(f"{API_PREFIX}/model-health")
    def model_health(
        window_days: int = 30,
        review_budget: int | None = None,
        svc: ScoringService = Depends(current_service),
        _: None = Depends(require_token),
    ) -> dict:
        """What the model is doing in production.

        The failures worth watching for here produce no errors: a lane that
        stopped loading, a rule returning indeterminate for every donation
        since a register went stale, an alert volume drifting past what the
        team can review. None of them appear in a request log.

        Recall is reported as unavailable with the reason. It needs a reviewed
        random sample of unflagged donations, which is an operational process
        and not something scoring events can be made to yield.
        """
        health = svc.model_health(window_days=window_days, review_budget=review_budget)
        return {
            "scored": health.scored,
            "window_days": health.window_days,
            "versions": health.versions,
            "lanes": [
                {
                    "lane": lane.lane,
                    "ran": lane.ran,
                    "did_not_run": lane.did_not_run,
                    "availability": lane.availability,
                    # Counted per distinct reason rather than summed: "no
                    # trained model is loaded" and "timed out" are different
                    # problems, and one availability figure hides which.
                    "reasons": lane.reasons,
                    "mean_contribution": lane.mean_contribution,
                }
                for lane in health.lanes
            ],
            "bands": health.bands,
            "rule_coverage": [
                {
                    "rule_id": rule_id,
                    "evaluated": evaluated,
                    "indeterminate": indeterminate,
                    "indeterminate_rate": (
                        indeterminate / (evaluated + indeterminate)
                        if evaluated + indeterminate
                        else 0.0
                    ),
                }
                for rule_id, evaluated, indeterminate in health.rule_coverage
            ],
            "alert_volume": {
                "flagged": health.alert_volume.flagged,
                "budget": health.alert_volume.budget,
                "over_budget": health.alert_volume.over_budget,
                "detail": health.alert_volume.describe(),
            },
            "degraded_share": health.degraded_share,
            "recall": health.recall,
            "recall_unavailable_reason": health.recall_unavailable_reason,
            # Ordered so a reader starts with what is actually wrong instead of
            # inferring it from a wall of figures.
            "concerns": list(health.concerns()),
        }

    @app.post(f"{API_PREFIX}/alerts/detect")
    def detect_alerts(
        svc: ScoringService = Depends(current_service),
        _: None = Depends(require_token),
    ) -> dict:
        """Run structural detection across the population.

        Separate from scoring because a cluster is not a property of any of its
        members: it becomes visible only once enough of it has arrived, and no
        per-donation call is the right moment to look for it.
        """
        alerts = svc.detect_group_alerts()
        return {
            "detected": len(alerts),
            "detected_at": (
                svc.alerts_detected_at.isoformat() if svc.alerts_detected_at else None
            ),
            "alerts": [alert.model_dump(mode="json", by_alias=True) for alert in alerts],
        }

    @app.get(f"{API_PREFIX}/alerts")
    def alerts(
        svc: ScoringService = Depends(current_service),
        _: None = Depends(require_token),
    ) -> dict:
        """Clusters as they stood at the last detection pass.

        Reports when detection last ran. A stale empty list and a genuinely
        clean population look identical otherwise, and only one of them means
        nothing was found.
        """
        found = svc.group_alerts
        return {
            "detected": len(found),
            "detected_at": (
                svc.alerts_detected_at.isoformat() if svc.alerts_detected_at else None
            ),
            "has_run": svc.alerts_detected_at is not None,
            "alerts": [alert.model_dump(mode="json", by_alias=True) for alert in found],
        }

    @app.get(f"{API_PREFIX}/explain/{{donation_id}}")
    def explain(
        donation_id: str,
        svc: ScoringService = Depends(current_service),
        _: None = Depends(require_token),
    ) -> dict:
        history = svc.history_for(donation_id)
        if not history:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"no scoring event recorded for {donation_id}",
            )
        return {
            "donation_id": donation_id,
            "events": [event.model_dump(mode="json") for event in history],
        }

    return app


app = create_app()
