"""Model card generation.

The card is assembled from the run's own manifest. What is tested is that it
cannot present a figure as something it is not — which is the failure the
previous published metrics represented, three times over.
"""

from __future__ import annotations

import pytest

from cakradana.governance import generate

BASE = {
    "trained_at": "2026-08-16T20:00:00+07:00",
    "config": {"seed": 1},
    "data": {"rows": 6694, "base_rate": 0.0617, "scale_pos_weight": 15.2},
    "versions": {"features": "features-abc", "rule_set": "rules-2026.07"},
    "splits": {
        "train": {"donations": 4898, "donors": 805},
        "calibration": {"donations": 986, "donors": 202},
        "test": {"donations": 810, "donors": 336},
    },
    "threshold": 0.027,
    "metrics": {
        "budget": 100,
        "precision_at_b": 0.54,
        "recall_at_b": 0.344,
        "lift_at_b": 0.10,
        "novel_finds": 6,
        "rule_baseline_finds": 63,
        "average_precision": 0.4486,
        "expected_calibration_error": 0.1474,
    },
    "label_basis": {"source": "synthetic", "is_human_confirmed": False},
    "ships": False,
}


def card(**overrides):
    manifest = {**BASE, **overrides}
    return generate(manifest, model_version="lgbm-test")


def flat(**overrides) -> str:
    """The card with line wrapping removed, for phrase assertions."""
    return " ".join(card(**overrides).split())


class TestHonesty:
    def test_generated_labels_are_declared_as_not_performance(self):
        """Generated labels say whether a model recovered planted patterns.
        Reporting that as detection is how a demonstration becomes a claim."""
        assert "not measurements of detection performance" in flat()

    def test_human_labels_carry_no_such_warning(self):
        text = flat(label_basis={"source": "human_confirmed", "is_human_confirmed": True})
        assert "not measurements of detection performance" not in text

    def test_an_unpromoted_model_says_so_first(self):
        text = card()
        assert "not promoted" in text
        assert text.index("not promoted") < text.index("Intended use")

    def test_a_promoted_model_states_its_incremental_yield(self):
        text = card(ships=True, metrics={**BASE["metrics"], "lift_at_b": 1.8})
        assert "Promoted" in text
        assert "1.80" in text

    def test_accuracy_is_absent_and_its_absence_explained(self):
        text = flat()
        assert "Accuracy and F1 are deliberately absent" in text
        assert "95% accuracy" in text

    def test_recall_is_qualified_by_what_was_reviewed(self):
        assert "only an estimate of true recall" in flat()


class TestScope:
    def test_the_card_refuses_the_claim_of_determining_an_offence(self):
        assert "does not determine that an offence occurred" in flat()

    def test_scoring_people_is_declared_out_of_scope(self):
        assert "not a person's character" in flat()

    def test_unevaluated_prohibitions_are_named(self):
        # Absence from a donation's findings is not evidence of compliance.
        assert "not evidence of compliance" in flat()


class TestProvenance:
    def test_every_version_is_recorded(self):
        text = card()
        assert "rules-2026.07" in text
        assert "features-abc" in text
        assert "lgbm-test" in text

    def test_the_split_strategy_and_its_limit_are_stated(self):
        text = flat()
        assert "grouped by donor" in text
        assert "does not measure" in text

    def test_the_real_class_distribution_is_reported(self):
        assert "not resampled to look balanced" in flat()
