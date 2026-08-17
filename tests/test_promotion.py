"""Promotion gates.

A model version existing is not a model version being in use. What is tested
here is that the second requires the first to pass every gate and to carry
somebody's name.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

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


def passing_report(**kwargs) -> GateReport:
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
