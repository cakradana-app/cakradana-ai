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
from cakradana.scoring.catalogue import ReasonCode, entry_for
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

        Two things follow from that and are enforced here rather than trusted.
        The wording comes from the reason-code catalogue, so what an analyst
        reads is the sentence that was reviewed. And a feature the catalogue
        says has no form a reader could check is passed over entirely, however
        much weight the model gave it — the ranking then continues down, so
        skipping one costs an explanation rather than suppressing it. A model
        input that cannot be stated is not made sayable by being important.
        """
        reasons: list[Reason] = []
        for name in self._importances:
            if len(reasons) == TOP_REASONS:
                break
            entry = entry_for(name.upper())
            if entry is None or not entry.analyst_facing:
                continue
            value = features.values.get(name)
            statement = _state(entry, value)
            if statement is None:
                continue
            reasons.append(
                Reason(
                    code=entry.code,
                    lane=Lane.CLASSIFIER,
                    weight=min(probability, 1.0),
                    statement=statement,
                    comparison=entry.comparison,
                    evidence_ref=f"donation:{features.donation_id}",
                )
            )
        if not reasons:
            reasons.append(
                Reason(
                    code="MODEL_SCORE",
                    lane=Lane.CLASSIFIER,
                    weight=min(probability, 1.0),
                    # "Ranked above most others on its combination of donor
                    # history, recipient pattern, and amount" described the
                    # model rather than the donation: those three are the
                    # inputs, and a position in an output distribution is not
                    # something an analyst can check against the record or
                    # quote into a case note.
                    #
                    # Restating the figure instead would fail the catalogue's
                    # own wording check, and rightly: a reason that repeats the
                    # score it is meant to explain explains nothing. What is
                    # left, and what is true, is that this lane contributed and
                    # nothing behind it can be checked — a limit of the
                    # evidence, stated the way HAS_UNRESOLVED_ENTITY and
                    # LANE_UNAVAILABLE state theirs.
                    statement=(
                        "The classifier lane contributed to this donation's "
                        "assessment, but none of the inputs it relied on most "
                        "carries wording a reader can check, so nothing here "
                        "can be checked against the record."
                    ),
                    evidence_ref=f"donation:{features.donation_id}",
                )
            )
        return tuple(reasons)


def _state(entry: ReasonCode, value: object) -> str | None:
    """Fill a catalogued sentence, or decline to say anything.

    None where there is nothing to say: a feature the vector never computed, and
    a boolean that came back false. Neither is an observation. Printing "false"
    beside a label would invite it to be read as one, and a null filled with
    anything at all would report a value nobody measured.
    """
    if entry.render == "when_true":
        return entry.statements[0] if value else None
    if value is None:
        return None
    return entry.statements[0].format(value=_render(value, entry.render))


def _render(value: object, kind: str) -> str:
    if kind == "rupiah" and isinstance(value, (int, float)):
        # Grouped with full stops, as rupiah is written in Indonesian. A figure
        # read under the other convention is wrong by three orders of magnitude.
        return f"Rp{int(value):,}".replace(",", ".")
    if kind == "share" and isinstance(value, (int, float)):
        return f"{value:.0%}"
    if isinstance(value, float):
        return f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)
