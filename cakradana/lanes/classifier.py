"""The classifier lane.

Loads a published artifact and scores a donation from the same feature
definitions that trained it. The feature order is checked on load rather than
trusted: a model fed its inputs in the wrong order does not fail, it produces
confident nonsense, and nothing downstream can tell the difference.

The lane refuses to run when a required feature is absent from the vector
altogether. That is different from a feature being null, which is a real state
the model was trained to handle — a donor's first donation genuinely has no
prior spread. A feature the vector never computed is a pipeline defect, and
scoring through it substitutes a fabricated input for a missing one.
"""

from __future__ import annotations

from cakradana.features import FeatureVector
from cakradana.rules.context import RuleContext
from cakradana.rules.engine import RuleEvaluation
from cakradana.scoring.composition import contribution_from, unavailable
from cakradana.scoring.result import Lane, LaneResult, Reason
from cakradana.training.registry import Artifact

#: Features whose importance is reported alongside a score. Kept to a handful
#: because a reason list that runs to forty entries is not an explanation.
TOP_REASONS = 3


class ClassifierLane:
    """Scores donations with a trained model."""

    name = Lane.CLASSIFIER

    def __init__(self, artifact: Artifact) -> None:
        self.artifact = artifact
        self._importances = self._rank_features()

    def _rank_features(self) -> tuple[str, ...]:
        model = self.artifact.model
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            return ()
        paired = sorted(
            zip(self.artifact.feature_names, importances),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return tuple(name for name, _ in paired)

    def evaluate(
        self,
        evaluation: RuleEvaluation,
        ctx: RuleContext,
        features: FeatureVector,
    ) -> LaneResult:
        missing = [
            name for name in self.artifact.feature_names if name not in features.values
        ]
        if missing:
            return unavailable(
                Lane.CLASSIFIER,
                (
                    "the feature vector is missing "
                    f"{len(missing)} input(s) the model requires "
                    f"({', '.join(missing[:3])}); scoring would substitute "
                    "values nobody computed"
                ),
            )

        probability = self._probability(features)
        reasons = self._reasons(features, probability)
        return contribution_from(
            Lane.CLASSIFIER, probability, reasons, probability=probability
        )

    def _probability(self, features: FeatureVector) -> float:
        import pandas as pd

        row = pd.DataFrame(
            [{name: features.values.get(name) for name in self.artifact.feature_names}],
            columns=list(self.artifact.feature_names),
        )
        for name in self.artifact.categorical_features:
            row[name] = row[name].astype("category")
        for name in self.artifact.feature_names:
            if name not in self.artifact.categorical_features:
                row[name] = pd.to_numeric(row[name], errors="coerce")

        raw = float(self.artifact.model.predict_proba(row)[:, 1][0])
        if self.artifact.calibrator is not None:
            # The calibrated value is what any published claim rests on: of
            # donations scored in a band, some measured share were confirmed
            # risky. The uncalibrated output supports no such statement.
            return float(self.artifact.calibrator.predict([raw])[0])
        return raw

    def _reasons(
        self, features: FeatureVector, probability: float
    ) -> tuple[Reason, ...]:
        """Name the inputs that carried the most weight, in plain terms.

        Model internals are not put in front of an analyst. A feature's
        importance is a property of the model, not of this donation, so the
        statement names what was observed rather than claiming the number
        caused the score.
        """
        reasons: list[Reason] = []
        for name in self._importances[:TOP_REASONS]:
            value = features.values.get(name)
            if value is None:
                continue
            reasons.append(
                Reason(
                    code=name.upper(),
                    lane=Lane.CLASSIFIER,
                    weight=min(probability, 1.0),
                    statement=f"{_phrase(name)}: {_render(value)}.",
                    evidence_ref=f"donation:{features.donation_id}",
                )
            )
        if not reasons:
            reasons.append(
                Reason(
                    code="MODEL_SCORE",
                    lane=Lane.CLASSIFIER,
                    weight=min(probability, 1.0),
                    statement=(
                        "The model ranked this donation above most others on "
                        "its combination of donor history, recipient pattern, "
                        "and amount."
                    ),
                    evidence_ref=f"donation:{features.donation_id}",
                )
            )
        return tuple(reasons)


def _phrase(name: str) -> str:
    return name.replace("_", " ")


def _render(value: object) -> str:
    if isinstance(value, float):
        return f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)
