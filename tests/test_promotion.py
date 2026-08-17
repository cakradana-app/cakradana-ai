"""Promotion gates.

A model version existing is not a model version being in use. What is tested
here is that the second requires the first to pass every gate and to carry
somebody's name.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from cakradana.evaluation.fairness import Cohort, assess
from cakradana.governance.promotion import (
    MAX_CALIBRATION_ERROR,
    MIN_LIFT,
    GateReport,
    PromotionRefused,
    current,
    evaluate_gates,
    promote,
    promoted_versions,
)
from cakradana.scoring.catalogue import entry_for, stateable_codes
from cakradana.scoring.result import ReviewStatus
from cakradana.scoring.review import ReviewDecision, ReviewLedger
from cakradana.training.registry import Artifact, ArtifactError


def manifest(**overrides) -> dict:
    base = {
        "trained_at": "2026-08-01T00:00:00+07:00",
        "config": {},
        "data": {"rows": 1000},
        "versions": {"features": "f-abc123", "rule_set": "rules-2026.07"},
        "splits": {"donor_overlap": 0},
        "threshold": 0.5,
        "metrics": {
            "precision_at_b": 0.62,
            "recall_at_b": 0.44,
            "lift_at_b": 1.8,
            "expected_calibration_error": 0.04,
        },
        "redundancy": {"rows": 800, "clean": True, "findings": []},
        "ships": True,
    }
    base.update(overrides)
    return base


def artifact(**overrides) -> Artifact:
    return Artifact(
        version="lgbm-2026.08.1",
        model=object(),
        calibrator=None,
        threshold=0.5,
        feature_names=("a",),
        categorical_features=(),
        manifest=manifest(**overrides),
    )


def even_handed_population() -> list[Cohort]:
    """A population on which no group is flagged in error more than another."""
    amounts = (5_000_000, 50_000_000, 500_000_000, 5_000_000_000)
    return [
        Cohort(
            donation_id=f"d-{index}",
            score=0.5,
            flagged=index % 7 == 0,
            affiliation=f"Partai{'AB'[index % 2]}",
            district=("Jakarta", "Surabaya")[index % 2],
            recipient_type="candidate" if index % 3 else "party",
            amount_idr=amounts[index % 4],
            reviewed=True,
            confirmed_risky=False,
        )
        for index in range(800)
    ]


def fully_reviewed() -> ReviewLedger:
    """A ledger in which every code has been read and accepted.

    Constructed rather than shipped: the real one is empty, and the gate that
    reads it is meant to block. This is what the other gates need in order to
    be tested on their own terms.
    """
    return ReviewLedger(
        ReviewDecision(
            code=code,
            status=ReviewStatus.VALIDATED,
            reviewer="analis@example.org",
            reviewed_at=datetime(2026, 8, 17, tzinfo=timezone.utc),
            note="reads as an observation",
            statements=entry_for(code).statements,
        )
        for code in stateable_codes()
    )


def passing_report(**kwargs) -> GateReport:
    kwargs.setdefault("fairness", assess(even_handed_population()))
    kwargs.setdefault("reason_reviews", fully_reviewed())
    return evaluate_gates(
        artifact(**kwargs.pop("artifact_overrides", {})),
        shadow_period_completed=True,
        golden_sets_passed=True,
        precision_floor=0.5,
        **kwargs,
    )


class TestGates:
    def test_a_sound_artifact_clears_every_evaluable_gate(self):
        report = passing_report()
        # G14 is checked against the directory at promotion time, so it is the
        # one gate that cannot pass on the manifest alone.
        outstanding = [g.gate for g in report.blocking]
        assert outstanding == ["G14"], report.describe()

    def test_lift_at_or_below_parity_blocks(self):
        """A model that cannot beat the rule engine at the operating budget
        does not ship, however good its AUC looks."""
        report = passing_report(artifact_overrides={"metrics": {**manifest()["metrics"], "lift_at_b": MIN_LIFT}})
        blocked = {g.gate for g in report.blocking}
        assert "G6" in blocked

    def test_donor_overlap_blocks(self):
        report = passing_report(artifact_overrides={"splits": {"donor_overlap": 3}})
        assert "G2" in {g.gate for g in report.blocking}

    def test_poor_calibration_blocks(self):
        report = passing_report(
            artifact_overrides={
                "metrics": {
                    **manifest()["metrics"],
                    "expected_calibration_error": MAX_CALIBRATION_ERROR + 0.01,
                }
            }
        )
        assert "G9" in {g.gate for g in report.blocking}

    def test_a_precision_regression_blocks(self):
        report = evaluate_gates(
            artifact(),
            shadow_period_completed=True,
            golden_sets_passed=True,
            precision_floor=0.9,
        )
        assert "G7" in {g.gate for g in report.blocking}

    def test_an_unevaluated_gate_blocks_like_a_failed_one(self):
        """Treating "could not check" as "fine" is how a promotion process
        comes to certify things nobody checked."""
        report = evaluate_gates(artifact())
        blocked = {g.gate for g in report.blocking}
        assert "G12" in blocked
        assert "G11" in blocked
        assert any(g.passed is None for g in report.results)

    def test_a_missing_metric_is_unevaluated_rather_than_failed(self):
        report = evaluate_gates(artifact(metrics={}))
        lift = next(g for g in report.results if g.gate == "G6")
        assert lift.passed is None
        assert "no Lift@B" in lift.detail

    def test_the_description_names_what_blocked(self):
        report = evaluate_gates(artifact())
        assert "blocked by" in report.describe()

    def test_an_absent_fairness_assessment_blocks(self):
        """A model whose output tracks party affiliation clears every other
        gate here. Nothing else in the report would notice."""
        report = evaluate_gates(
            artifact(),
            shadow_period_completed=True,
            golden_sets_passed=True,
            precision_floor=0.5,
        )
        fairness = next(g for g in report.results if g.gate == "G13")
        assert fairness.passed is None
        assert fairness.blocks

    def test_a_measured_disparity_blocks_and_names_the_dimension(self):
        biased = [
            *(
                Cohort(
                    donation_id=f"a-{i}",
                    score=0.5,
                    flagged=i < 80,
                    affiliation="PartaiA",
                    district="Jakarta",
                    recipient_type="candidate",
                    amount_idr=5_000_000,
                    reviewed=True,
                )
                for i in range(200)
            ),
            *(
                Cohort(
                    donation_id=f"b-{i}",
                    score=0.5,
                    flagged=i < 20,
                    affiliation="PartaiB",
                    district="Surabaya",
                    recipient_type="candidate",
                    amount_idr=5_000_000,
                    reviewed=True,
                )
                for i in range(200)
            ),
        ]
        report = passing_report(fairness=assess(biased))
        gate = next(g for g in report.results if g.gate == "G13")
        assert gate.passed is False
        assert "affiliation" in gate.detail

    def test_a_duplicated_feature_blocks(self):
        report = passing_report(
            artifact_overrides={
                "redundancy": {
                    "rows": 800,
                    "clean": False,
                    "findings": [
                        {
                            "kind": "identical",
                            "columns": ["degree_centrality_sender", "sender_out_degree"],
                            "detail": "identical in every row",
                        }
                    ],
                }
            }
        )
        gate = next(g for g in report.results if g.gate == "G8")
        assert gate.passed is False
        assert "degree_centrality_sender" in gate.detail

    def test_a_constant_column_alone_does_not_block(self):
        """A feature whose inputs are absent in this dataset is a data problem
        worth reporting, but it is not a reason to refuse the model."""
        report = passing_report(
            artifact_overrides={
                "redundancy": {
                    "rows": 800,
                    "clean": False,
                    "findings": [
                        {
                            "kind": "constant",
                            "columns": ["has_unresolved_entity"],
                            "detail": "every row holds False",
                        }
                    ],
                }
            }
        )
        assert next(g for g in report.results if g.gate == "G8").passed is True

    def test_a_run_that_never_checked_for_duplicates_blocks(self):
        report = passing_report(artifact_overrides={"redundancy": None})
        gate = next(g for g in report.results if g.gate == "G8")
        assert gate.passed is None
        assert gate.blocks

    def test_an_even_handed_model_clears_the_fairness_gate(self):
        report = passing_report()
        gate = next(g for g in report.results if g.gate == "G13")
        assert gate.passed is True


class TestPromotion:
    @pytest.fixture
    def registry(self, tmp_path):
        directory = tmp_path / "lgbm-2026.08.1"
        directory.mkdir()
        (directory / "MODEL_CARD.md").write_text("# card", encoding="utf-8")
        return tmp_path

    def clear(self) -> GateReport:
        report = passing_report()
        # G14 is satisfied by the directory, which promote() checks itself.
        return GateReport(tuple(g for g in report.results if g.gate != "G14"))

    def test_a_clear_report_promotes_and_records_the_approver(self, registry):
        record = promote(
            "lgbm-2026.08.1",
            approved_by="ml-lead@example.org",
            report=self.clear(),
            now=datetime(2026, 8, 17, tzinfo=timezone.utc),
            root=registry,
        )
        assert record.approved_by == "ml-lead@example.org"
        written = json.loads(
            (registry / "lgbm-2026.08.1" / "PROMOTION.json").read_text()
        )
        assert written["approved_by"] == "ml-lead@example.org"

    def test_nothing_promotes_itself(self, registry):
        with pytest.raises(PromotionRefused, match="name the person"):
            promote(
                "lgbm-2026.08.1", approved_by="", report=self.clear(), root=registry
            )

    def test_a_blocking_gate_refuses_and_says_which(self, registry):
        report = evaluate_gates(artifact())
        with pytest.raises(PromotionRefused, match="G12"):
            promote(
                "lgbm-2026.08.1",
                approved_by="ml-lead@example.org",
                report=report,
                root=registry,
            )

    def test_a_version_without_a_model_card_is_refused(self, tmp_path):
        """A regulator reading about this system has no other way to learn what
        it does not detect."""
        (tmp_path / "lgbm-2026.08.1").mkdir()
        with pytest.raises(PromotionRefused, match="model card"):
            promote(
                "lgbm-2026.08.1",
                approved_by="ml-lead@example.org",
                report=self.clear(),
                root=tmp_path,
            )

    def test_an_unknown_version_is_an_error(self, tmp_path):
        with pytest.raises(ArtifactError):
            promote(
                "nothing-here",
                approved_by="ml-lead@example.org",
                report=self.clear(),
                root=tmp_path,
            )

    def test_there_is_no_force_argument(self):
        """A gate that can be waived under pressure is a comment, and the
        pressure is exactly when it matters."""
        import inspect

        parameters = inspect.signature(promote).parameters
        assert not any(
            name in parameters for name in ("force", "override", "skip_gates")
        )


class TestCurrent:
    def test_nothing_is_in_service_until_something_is_promoted(self, tmp_path):
        """Falling back to the newest artifact would serve an unpromoted model
        because it happened to sort last, which is what the record prevents."""
        (tmp_path / "lgbm-2026.08.1").mkdir()
        assert current(root=tmp_path) is None
        assert promoted_versions(tmp_path) == ()

    def test_the_promoted_version_is_readable_back(self, tmp_path):
        directory = tmp_path / "lgbm-2026.08.1"
        directory.mkdir()
        (directory / "MODEL_CARD.md").write_text("# card", encoding="utf-8")
        report = GateReport(
            tuple(g for g in passing_report().results if g.gate != "G14")
        )
        promote(
            "lgbm-2026.08.1",
            approved_by="ml-lead@example.org",
            report=report,
            note="shadow ran for two weeks",
            root=tmp_path,
        )
        live = current(root=tmp_path)
        assert live is not None
        assert live.version == "lgbm-2026.08.1"
        assert live.note == "shadow ran for two weeks"


class TestReasonWordingGate:
    """G10. A score nobody can explain is not a score anybody can defend.

    The gate reads a shipped record rather than taking a caller's word for it,
    and it fails today because nothing in that record says anybody has read the
    sentences the system puts in front of an analyst.
    """

    def gate(self, report: GateReport) -> object:
        return next(g for g in report.results if g.gate == "G10")

    def test_nothing_reviewed_blocks(self):
        report = passing_report(reason_reviews=ReviewLedger())
        assert self.gate(report).passed is False
        assert self.gate(report).blocks

    def test_the_shipped_state_of_this_system_blocks(self):
        """Asserted against the real ledger, not a fixture.

        It blocked when nothing had been read. It blocks now that everything
        has, because five wordings were found misleading and are still
        emitted — which is the harder case and the one worth pinning: a gate
        that counted reviews rather than weighing them would have opened the
        moment the last decision was recorded, whatever those decisions said.

        The rejected codes are named in the detail so that whoever reads a
        blocked gate learns which five sentences to rewrite.
        """
        detail = self.gate(evaluate_gates(artifact())).detail
        assert self.gate(evaluate_gates(artifact())).passed is False
        assert "misleading and still emitted" in detail
        for code in ("MODEL_SCORE", "UNUSUAL_COMBINATION", "PASS_THROUGH"):
            assert code in detail

    def test_a_wording_found_misleading_blocks_even_if_everything_else_is_read(self):
        ledger = ReviewLedger(
            [
                *fully_reviewed().decisions,
                ReviewDecision(
                    code="LANE_UNAVAILABLE",
                    status=ReviewStatus.REJECTED,
                    reviewer="analis@example.org",
                    reviewed_at=datetime(2026, 8, 18, tzinfo=timezone.utc),
                    note="names the lane but not what the reader should do next",
                    statements=entry_for("LANE_UNAVAILABLE").statements,
                ),
            ]
        )
        report = passing_report(reason_reviews=ledger)
        assert self.gate(report).passed is False
        assert "misleading" in self.gate(report).detail

    def test_a_missing_ledger_is_not_evaluated_rather_than_passed(self, monkeypatch):
        monkeypatch.setattr(
            "cakradana.governance.promotion.default_ledger", lambda: None
        )
        report = evaluate_gates(artifact())
        assert self.gate(report).passed is None
        assert self.gate(report).blocks

    def test_every_wording_read_and_accepted_clears_it(self):
        report = passing_report(reason_reviews=fully_reviewed())
        assert self.gate(report).passed is True

    def test_the_gate_cannot_be_waived(self, tmp_path):
        """There is no force argument anywhere in this module, and a report
        carrying a blocking gate refuses however it was assembled."""
        directory = tmp_path / "lgbm-2026.08.1"
        directory.mkdir()
        (directory / "MODEL_CARD.md").write_text("# card", encoding="utf-8")
        report = GateReport(
            tuple(
                g
                for g in passing_report(reason_reviews=ReviewLedger()).results
                if g.gate != "G14"
            )
        )
        with pytest.raises(PromotionRefused, match="G10"):
            promote(
                "lgbm-2026.08.1",
                approved_by="ml-lead@example.org",
                report=report,
                root=tmp_path,
            )
