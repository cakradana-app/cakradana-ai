"""Model artifacts and their provenance.

An artifact is the model, the calibration mapping, the operating threshold, the
feature definitions it expects, the rule set it was trained beside, and the
manifest describing the run that produced all of it. These travel together
because they are only meaningful together: a model loaded with the wrong
feature order produces confident nonsense, and a threshold separated from the
run that chose it is a number with no justification.

Versions are never overwritten. Retraining writes a new version, so a score
recorded last quarter can still be reproduced by loading the artifact that
produced it.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from cakradana.training.pipeline import TrainingResult

ARTIFACT_ROOT = Path("artifacts")

MODEL_FILE = "model.joblib"
CALIBRATOR_FILE = "calibrator.joblib"
MANIFEST_FILE = "manifest.json"
FEATURES_FILE = "features.json"
MODEL_CARD_FILE = "MODEL_CARD.md"


class ArtifactError(RuntimeError):
    """Raised when an artifact is missing or internally inconsistent."""


@dataclass(frozen=True)
class Artifact:
    """A loaded model version, ready to serve."""

    version: str
    model: object
    calibrator: object | None
    threshold: float
    feature_names: tuple[str, ...]
    categorical_features: tuple[str, ...]
    manifest: dict

    @property
    def feature_set_version(self) -> str:
        return self.manifest.get("versions", {}).get("features", "unknown")

    @property
    def rule_set_version(self) -> str:
        return self.manifest.get("versions", {}).get("rule_set", "unknown")

    @property
    def shipped_on_merit(self) -> bool:
        """Whether this artifact met the bar for adding value over the rules."""
        return bool(self.manifest.get("ships"))


def save(
    result: TrainingResult,
    version: str,
    *,
    feature_names: Sequence[str],
    categorical_features: Sequence[str],
    root: Path = ARTIFACT_ROOT,
) -> Path:
    """Write a model version. Refuses to overwrite an existing one."""
    import joblib

    directory = root / version
    if directory.exists():
        raise ArtifactError(
            f"version {version!r} already exists; retraining writes a new "
            f"version so that earlier scores stay reproducible"
        )
    directory.mkdir(parents=True)

    joblib.dump(result.model, directory / MODEL_FILE)
    if result.calibrator is not None:
        joblib.dump(result.calibrator, directory / CALIBRATOR_FILE)

    (directory / FEATURES_FILE).write_text(
        json.dumps(
            {
                "feature_names": list(feature_names),
                "categorical_features": list(categorical_features),
                "threshold": result.threshold,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (directory / MANIFEST_FILE).write_text(
        json.dumps(result.manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    # The card is generated from the manifest rather than written alongside it,
    # so it cannot claim a figure the run did not produce. A card composed by
    # hand records what somebody believed at the time.
    from cakradana.governance.model_card import write as write_card

    write_card(result.manifest, directory / MODEL_CARD_FILE, model_version=version)
    return directory


def load(version: str, *, root: Path = ARTIFACT_ROOT) -> Artifact:
    """Load a model version.

    Every part must be present. A service that starts with a missing artifact
    and scores anyway is worse than one that refuses to start, because the
    failure is invisible to whoever reads its output.
    """
    import joblib

    directory = root / version
    if not directory.is_dir():
        raise ArtifactError(f"no artifact at {directory}")

    for required in (MODEL_FILE, FEATURES_FILE, MANIFEST_FILE):
        if not (directory / required).is_file():
            raise ArtifactError(f"{directory} is missing {required}")

    spec = json.loads((directory / FEATURES_FILE).read_text(encoding="utf-8"))
    manifest = json.loads((directory / MANIFEST_FILE).read_text(encoding="utf-8"))
    calibrator_path = directory / CALIBRATOR_FILE

    return Artifact(
        version=version,
        model=joblib.load(directory / MODEL_FILE),
        calibrator=(
            joblib.load(calibrator_path) if calibrator_path.is_file() else None
        ),
        threshold=float(spec["threshold"]),
        feature_names=tuple(spec["feature_names"]),
        categorical_features=tuple(spec["categorical_features"]),
        manifest=manifest,
    )


def versions(root: Path = ARTIFACT_ROOT) -> tuple[str, ...]:
    if not root.is_dir():
        return ()
    return tuple(
        sorted(p.name for p in root.iterdir() if (p / MANIFEST_FILE).is_file())
    )


def latest(root: Path = ARTIFACT_ROOT) -> Artifact:
    """The newest artifact on disk.

    Newest, not live. Serving is loaded from `governance.promotion.current`,
    which reads the version somebody approved; a directory sorting last is not
    a decision anybody made. This exists for inspecting what training produced.
    """
    available = versions(root)
    if not available:
        raise ArtifactError(f"no model versions published under {root}")
    return load(available[-1], root=root)


def load_promoted(root: Path = ARTIFACT_ROOT) -> Artifact | None:
    """The version approved for service, or None.

    None rather than a fallback to the newest. A deployment with nothing
    promoted should run the rules alone — which is a working system — instead
    of quietly serving whatever was trained most recently.
    """
    from cakradana.governance.promotion import current

    promotion = current(root)
    return load(promotion.version, root=root) if promotion else None


def remove(version: str, *, root: Path = ARTIFACT_ROOT) -> None:
    """Delete a version.

    Provided for cleaning up runs that were never promoted. Removing a version
    that has served traffic destroys the ability to explain what it produced.
    """
    directory = root / version
    if directory.is_dir():
        shutil.rmtree(directory)
