"""Assembling training data.

Rows are produced by replaying donations in the order the system learned of
them, so every feature value is one that could have been served at the moment
the donation arrived. This is the same code path serving uses; there is no
separate training-time computation that could drift from it.

Labels come from the behavioural heuristics, at reduced weight. They are
hypotheses about intent inferred from structure, not observations, and the
weighting is what keeps them from being treated as ground truth. Statutory
outcomes are deliberately not labels: a model trained on those could only
relearn arithmetic it was already handed, and on donations the statute had
already cleared it would return negatives by construction.

The class balance is left as it is. Resampling to look even makes class
weighting inert and produces precision figures that do not transfer to a
population where the pattern is rare.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Mapping, Sequence

from cakradana.features import FeatureService, FeatureValue
from cakradana.history import DonationStore
from cakradana.rules import RuleEngine
from cakradana.schema import Donation, Entity


@dataclass(frozen=True)
class TrainingRow:
    """One donation as the model sees it."""

    donation_id: str
    donor_id: str | None
    occurred_at: datetime
    features: Mapping[str, FeatureValue]
    label: int
    #: Confidence in the label. Heuristic positives enter below full weight.
    weight: float
    #: Whether any behavioural heuristic fired. Needed to measure what the
    #: model finds that the rules did not.
    rule_flagged: bool
    typologies: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrainingData:
    rows: tuple[TrainingRow, ...]
    feature_set_version: str
    rule_set_version: str

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def positives(self) -> int:
        return sum(r.label for r in self.rows)

    @property
    def base_rate(self) -> float:
        return self.positives / len(self.rows) if self.rows else 0.0

    def scale_pos_weight(self) -> float:
        """Ratio of negatives to positives.

        Meaningful only because the data keeps its real imbalance. On the
        previous half-and-half dataset this evaluated to almost exactly one,
        so the setting it feeds did nothing at all.
        """
        positives = self.positives
        if positives == 0:
            return 1.0
        return (len(self.rows) - positives) / positives


def build_training_data(
    store: DonationStore,
    engine: RuleEngine,
    features: FeatureService,
    *,
    entities: Mapping[str, Entity] | None = None,
    negative_weight: float = 1.0,
) -> TrainingData:
    """Replay a store into labelled rows."""
    rows: list[TrainingRow] = []
    for donation, view in store.replay():
        ctx = features.context_for(donation, view, entities=entities)
        evaluation = engine.evaluate(
            donation, view, now=ctx.now, entities=entities
        )
        vector = features.compute_from_context(ctx)

        signals = evaluation.behavioural_signals
        label = 1 if signals else 0
        weight = (
            evaluation.tier2_label_weight() if signals else negative_weight
        )
        rows.append(
            TrainingRow(
                donation_id=donation.donation_id,
                donor_id=donation.sender_ref.entity_id,
                occurred_at=donation.occurred_at,
                features=vector.values,
                label=label,
                weight=weight,
                rule_flagged=bool(signals),
                typologies=tuple(
                    s.typology for s in signals if s.typology is not None
                ),
            )
        )

    return TrainingData(
        rows=tuple(rows),
        feature_set_version=features.version,
        rule_set_version=engine.ruleset.version,
    )


def to_frame(rows: Sequence[TrainingRow], features: FeatureService):
    """Build a dataframe the gradient booster can consume directly.

    Categoricals are typed as such rather than integer-encoded, and missing
    values stay missing. The booster splits on absence natively, which is the
    correct treatment: a donor with no history has no mean interval between
    donations, and any number substituted there is a claim the data does not
    support.
    """
    import pandas as pd

    frame = pd.DataFrame(
        [dict(row.features) for row in rows], columns=list(features.names)
    )
    for name in features.categorical:
        frame[name] = frame[name].astype("category")
    for name in features.numeric:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    return frame


def labels_of(rows: Iterable[TrainingRow]) -> list[int]:
    return [row.label for row in rows]


def weights_of(rows: Iterable[TrainingRow]) -> list[float]:
    return [row.weight for row in rows]
