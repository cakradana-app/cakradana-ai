"""Training a classifier and deciding whether it is worth shipping.

The pipeline ends in a judgement, not a model. Training always produces
something; whether that something adds anything to the rules already in place
is a separate question, and it is answered here rather than assumed.

Calibration happens on its own split. Fitting it on the training data
reproduces the model's own overconfidence, and fitting it on the test data
spends the only honest performance estimate available.

Everything the run depends on is recorded — data, feature definitions, rules,
parameters, splits, and the resulting metrics — because a score that cannot be
traced back to the run that produced it cannot be explained six months later,
and this system's outputs feed processes where that question gets asked.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Mapping, Sequence

from cakradana.evaluation.metrics import (
    BudgetMetrics,
    CalibrationReport,
    Scored,
    average_precision,
    calibration_error,
    lift_at_budget,
    select_threshold,
)
from cakradana.evaluation.splits import SplitSet, donor_cohort_split
from cakradana.features import FeatureService
from cakradana.history import DonationStore
from cakradana.rules import RuleEngine
from cakradana.schema import Entity
from cakradana.training.dataset import (
    TrainingData,
    TrainingRow,
    build_training_data,
    to_frame,
)


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 20260816
    num_leaves: int = 31
    learning_rate: float = 0.05
    n_estimators: int = 400
    min_child_samples: int = 20
    #: Floor on how many genuinely clean donations are left alone, used when
    #: choosing an operating threshold.
    min_recall_not_risky: float = 0.70
    #: Donations a team can review in the period the metrics describe.
    review_budget: int = 100


@dataclass(frozen=True)
class LabelBasis:
    """What the evaluation labels actually are.

    Carried with every result because it changes what the numbers mean.
    Generated labels describe whether a model recovered patterns that were
    planted for it; only human judgement says anything about real detection,
    and reporting one as the other is how a demo becomes a claim.
    """

    source: str
    is_human_confirmed: bool

    @property
    def reportable_as_system_performance(self) -> bool:
        return self.is_human_confirmed


SYNTHETIC_LABELS = LabelBasis(source="synthetic", is_human_confirmed=False)
HUMAN_LABELS = LabelBasis(source="human_confirmed", is_human_confirmed=True)


@dataclass
class TrainingResult:
    model: object
    calibrator: object | None
    threshold: float
    metrics: BudgetMetrics
    calibration: CalibrationReport
    average_precision: float
    label_basis: LabelBasis
    splits: Mapping[str, object]
    manifest: dict[str, object] = field(default_factory=dict)

    @property
    def should_ship(self) -> bool:
        """Whether the classifier earns a place beside the rules.

        Lift at or below parity means the rules alone surface as much as the
        model adds. They are cheaper to run, already explainable, and already
        built, so the model does not ship and the rules run on their own.
        """
        return self.metrics.model_earns_its_place

    def summary(self) -> str:
        verdict = (
            "adds incremental detection over the rules"
            if self.should_ship
            else "adds nothing over the rules; do not ship"
        )
        caveat = (
            ""
            if self.label_basis.reportable_as_system_performance
            else "  [measured against generated labels, not system performance]"
        )
        return f"{self.metrics.describe()} -> {verdict}{caveat}"


def train(
    store: DonationStore,
    engine: RuleEngine,
    features: FeatureService,
    *,
    truth: Mapping[str, str] | None = None,
    label_basis: LabelBasis = SYNTHETIC_LABELS,
    entities: Mapping[str, Entity] | None = None,
    config: TrainingConfig | None = None,
) -> TrainingResult:
    """Train, calibrate, choose a threshold, and measure.

    ``truth`` supplies the evaluation labels. It is deliberately separate from
    the heuristic labels used for fitting: measuring a model against the rules
    it was trained on measures only how well it memorised them.
    """
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression

    config = config or TrainingConfig()
    data = build_training_data(store, engine, features, entities=entities)
    if not data.rows:
        raise ValueError("no donations to train on")

    by_id = {row.donation_id: row for row in data.rows}
    donations = [d for d, _ in store.replay()]
    splits = donor_cohort_split(donations)

    train_rows = _rows_for(splits.train.donations, by_id)
    calibration_rows = _rows_for(splits.calibration.donations, by_id)
    test_rows = _rows_for(splits.test.donations, by_id)
    if not train_rows or not test_rows:
        raise ValueError("splitting left a partition empty")

    model = lgb.LGBMClassifier(
        objective="binary",
        num_leaves=config.num_leaves,
        learning_rate=config.learning_rate,
        n_estimators=config.n_estimators,
        min_child_samples=config.min_child_samples,
        scale_pos_weight=data.scale_pos_weight(),
        random_state=config.seed,
        verbose=-1,
    )
    model.fit(
        to_frame(train_rows, features),
        [r.label for r in train_rows],
        sample_weight=[r.weight for r in train_rows],
        categorical_feature=list(features.categorical),
    )

    calibrator = None
    if calibration_rows:
        raw = model.predict_proba(to_frame(calibration_rows, features))[:, 1]
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw, [r.label for r in calibration_rows])

    def probabilities(rows: Sequence[TrainingRow]):
        raw = model.predict_proba(to_frame(rows, features))[:, 1]
        return calibrator.predict(raw) if calibrator is not None else raw

    scored_calibration = _scored(calibration_rows, probabilities, truth)
    threshold = (
        select_threshold(
            scored_calibration, min_recall_not_risky=config.min_recall_not_risky
        )
        if scored_calibration
        else 0.5
    )

    scored_test = _scored(test_rows, probabilities, truth)
    metrics = lift_at_budget(scored_test, config.review_budget)

    result = TrainingResult(
        model=model,
        calibrator=calibrator,
        threshold=float(threshold),
        metrics=metrics,
        calibration=calibration_error(scored_test),
        average_precision=average_precision(scored_test),
        label_basis=label_basis,
        splits=splits.summary(),
    )
    result.manifest = _manifest(result, data, config, splits, features, engine)
    return result


def _rows_for(donations, by_id) -> list[TrainingRow]:
    return [by_id[d.donation_id] for d in donations if d.donation_id in by_id]


def _scored(rows, probabilities, truth) -> list[Scored]:
    if not rows:
        return []
    scores = probabilities(rows)
    return [
        Scored(
            donation_id=row.donation_id,
            score=float(score),
            confirmed_risky=bool(truth and row.donation_id in truth),
            rule_flagged=row.rule_flagged,
        )
        for row, score in zip(rows, scores)
    ]


def _manifest(
    result: TrainingResult,
    data: TrainingData,
    config: TrainingConfig,
    splits: SplitSet,
    features: FeatureService,
    engine: RuleEngine,
) -> dict[str, object]:
    return {
        "trained_at": datetime.now().astimezone().isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "config": asdict(config),
        "data": {
            "rows": len(data),
            "heuristic_positives": data.positives,
            "base_rate": round(data.base_rate, 4),
            "scale_pos_weight": round(data.scale_pos_weight(), 4),
        },
        "versions": {
            "features": features.version,
            "rule_set": engine.ruleset.version,
        },
        "splits": splits.summary(),
        "threshold": result.threshold,
        "metrics": {
            "budget": result.metrics.budget,
            "precision_at_b": round(result.metrics.precision_at_b, 4),
            "recall_at_b": round(result.metrics.recall_at_b, 4),
            "lift_at_b": round(result.metrics.lift_at_b, 4),
            "novel_finds": result.metrics.novel_finds,
            "rule_baseline_finds": result.metrics.rule_baseline_finds,
            "average_precision": round(result.average_precision, 4),
            "expected_calibration_error": round(
                result.calibration.expected_calibration_error, 4
            ),
        },
        "label_basis": asdict(result.label_basis),
        "ships": result.should_ship,
    }


def manifest_json(result: TrainingResult) -> str:
    return json.dumps(result.manifest, indent=2, sort_keys=True)
