"""The anomaly lane.

Looks for donations that are unusual without matching any known pattern. Its
purpose is coverage of what the typologies do not describe: the catalogue was
written from patterns people have already seen, and the ones nobody has
described yet are exactly the ones a rule cannot catch.

It is the least precise lane by design and is capped accordingly. Unusual is
not suspicious — a legitimate large donation from a first-time corporate donor
is unusual every time — so this lane produces candidates for a human to look
at, and never a finding.

It runs over the population the statutory rules have already cleared. Donations
that breach a limit are reported as breaches; asking whether they are also
statistically odd adds nothing an analyst can use.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Sequence

from cakradana.features import FeatureService, FeatureVector
from cakradana.rules.context import RuleContext
from cakradana.rules.engine import RuleEvaluation
from cakradana.scoring.composition import contribution_from, unavailable
from cakradana.scoring.result import Lane, LaneResult, Reason


@dataclass(frozen=True)
class AnomalyModel:
    """A fitted outlier detector and the features it expects."""

    detector: object
    feature_names: tuple[str, ...]
    #: Per-column stand-ins for absent values, fixed when the model was fitted.
    #: Recomputing them at scoring time would substitute different values than
    #: training did, which is the train/serve divergence this project exists to
    #: rule out — and scoring one donation would derive a "median" from that
    #: donation alone.
    fill_values: tuple[float, ...]
    #: Score below which a donation is treated as ordinary. Chosen from the
    #: fitted population rather than assumed, so the lane surfaces a bounded
    #: share of traffic instead of whatever a default happens to yield.
    cutoff: float


def fit(
    vectors: Sequence[FeatureVector],
    features: FeatureService,
    *,
    contamination: float = 0.03,
    seed: int = 20260816,
) -> AnomalyModel:
    """Fit an outlier detector on donations the rules did not flag.

    ``contamination`` states the share of the population expected to be
    unusual. It is a property of the data rather than a target, and it is set
    to match the risk prevalence the data was built around instead of being
    left at a library default that would surface an arbitrary fraction.
    """
    import numpy as np
    from sklearn.ensemble import IsolationForest

    names = tuple(features.numeric)
    raw = _raw_matrix(vectors, names)
    if raw.shape[0] < 20:
        raise ValueError(
            "too few donations to characterise what is ordinary; an outlier "
            "detector fitted on a handful of records describes the handful"
        )

    # A column can be absent for every donation in a dataset — a deadline
    # feature where no deadline is configured, for instance. That is a real
    # state rather than an error, so the empty-slice warning is suppressed and
    # the column is filled with zero, which is as informative as anything else
    # about a quantity nothing could compute.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        medians = np.nanmedian(raw, axis=0) if raw.size else np.zeros(len(names))
    medians = np.where(np.isnan(medians), 0.0, medians)
    matrix = np.where(np.isnan(raw), medians, raw)

    detector = IsolationForest(
        contamination=contamination, random_state=seed, n_estimators=200
    )
    detector.fit(matrix)

    # The cutoff is read off the fitted population so the lane's output volume
    # is known in advance rather than discovered in production.
    scores = -detector.score_samples(matrix)
    cutoff = float(np.quantile(scores, 1.0 - contamination))
    return AnomalyModel(
        detector=detector,
        feature_names=names,
        fill_values=tuple(float(v) for v in medians),
        cutoff=cutoff,
    )


def _raw_matrix(vectors: Sequence[FeatureVector], names: Sequence[str]):
    """Numeric matrix with absent values left as missing."""
    import numpy as np

    return np.array(
        [
            [_as_float(vector.values.get(name)) for name in names]
            for vector in vectors
        ],
        dtype=float,
    )


def _filled_matrix(
    vectors: Sequence[FeatureVector], model: AnomalyModel
):
    """Numeric matrix with absent values replaced by the fitted stand-ins.

    This is the one place a substituted value is acceptable: the detector
    cannot consume missing values at all, and the substitution never reaches a
    figure presented as a measurement. The fitted median is used rather than
    zero, because zero is a meaningful amount and a meaningful count, and
    putting it where a value is unknown invents an observation.
    """
    import numpy as np

    raw = _raw_matrix(vectors, model.feature_names)
    fill = np.array(model.fill_values, dtype=float)
    return np.where(np.isnan(raw), fill, raw)


def _as_float(value: object) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return float("nan")


class AnomalyLane:
    """Surfaces donations that are unusual without matching a known pattern."""

    name = Lane.ANOMALY

    def __init__(self, model: AnomalyModel) -> None:
        self.model = model

    def evaluate(
        self,
        evaluation: RuleEvaluation,
        ctx: RuleContext,
        features: FeatureVector,
    ) -> LaneResult:
        if evaluation is not None and evaluation.legal_findings:
            # The donation already carries a statutory finding. Reporting that
            # it is also statistically unusual adds nothing an analyst can act
            # on and spends part of a capped budget saying so.
            return unavailable(
                Lane.ANOMALY,
                "not evaluated: the donation already carries a legal finding",
            )

        score = self._score(features)
        if score < self.model.cutoff:
            return contribution_from(Lane.ANOMALY, 0.0, ())

        # Intensity is how far past the cutoff the donation sits, saturating
        # quickly: this lane ranks candidates for attention, and a precise
        # ordering among extreme outliers is not a distinction it can support.
        intensity = min((score - self.model.cutoff) / max(self.model.cutoff, 1e-9), 1.0)
        reason = Reason(
            code="UNUSUAL_COMBINATION",
            lane=Lane.ANOMALY,
            weight=round(intensity, 3),
            # "Unlike the donations around it, without matching any known
            # pattern" named no quantity, no comparison set and no threshold,
            # so an analyst could not tell a near miss from an extreme and had
            # nothing to disagree with.
            #
            # An isolation forest cannot say which quantities put a donation
            # outside the range, and stating the raw outlier value instead
            # would restate the score — which the catalogue's own wording check
            # rejects, because a reason that repeats the number it is meant to
            # explain explains nothing. So the sentence says what happened and
            # then says what cannot be known, in the manner of LANE_UNAVAILABLE
            # and HAS_UNRESOLVED_ENTITY: a contribution nobody can check is a
            # limit of the evidence, and naming it is the useful thing to do
            # with it.
            statement=(
                "The anomaly lane placed this donation outside the range it "
                "was fitted to treat as ordinary. It does not identify which "
                "quantities put it there, so nothing here can be checked "
                "against the record."
            ),
            comparison=(
                "Most donations sit well inside the range this lane treats as "
                "ordinary."
            ),
            evidence_ref=f"donation:{features.donation_id}",
        )
        return contribution_from(Lane.ANOMALY, intensity, (reason,))

    def _score(self, features: FeatureVector) -> float:
        matrix = _filled_matrix([features], self.model)
        return float(-self.model.detector.score_samples(matrix)[0])
