"""Whether a trained model may serve, and who said so.

A model version existing is not the same as a model version being in use, and
the gap between them is where this system's accountability sits. Training
produces artifacts continuously; promotion is a decision, made once, by somebody
whose name goes on it.

Two things are enforced here.

**The gates are checks, not warnings.** Each returns a fact about the artifact
in front of it, and a failure blocks promotion rather than annotating it. A gate
that can be overridden by whoever is in a hurry is a comment.

**Nothing promotes itself.** There is no code path that marks a version live
without an approver, because the point of the gate is that a person looked at
the numbers and accepted responsibility for them. The gate that matters most is
Lift@B: a model that cannot beat the rule engine at the operating budget does
not ship, however good its AUC looks, and the rules alone remain a perfectly
respectable product.

Gates that cannot be evaluated from the artifact — a completed shadow period, a
compliance sign-off — are reported as outstanding rather than assumed passed.
An unevaluated gate and a passed gate are different states, and only one of them
is evidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Sequence

from cakradana.evaluation.fairness import FairnessReport
from cakradana.scoring.review import ReviewLedger, default_ledger
from cakradana.training.registry import ARTIFACT_ROOT, Artifact, ArtifactError

PROMOTION_FILE = "PROMOTION.json"

#: Calibration error above which the score's stated meaning is not supported.
MAX_CALIBRATION_ERROR = 0.10

#: Below this, the classifier adds nothing the rules do not already surface.
MIN_LIFT = 1.0


@dataclass(frozen=True)
class GateResult:
    """One gate, and what it found."""

    gate: str
    title: str
    #: None when the gate could not be evaluated from what is available.
    passed: bool | None
    detail: str

    @property
    def blocks(self) -> bool:
        """Anything not positively passing blocks.

        An unevaluated gate blocks too. Treating "could not check" as "fine"
        is how a promotion process comes to certify things nobody checked.
        """
        return self.passed is not True

    def describe(self) -> str:
        mark = {True: "pass", False: "FAIL", None: "not evaluated"}[self.passed]
        return f"{self.gate:<4} {mark:<14} {self.title} — {self.detail}"


@dataclass(frozen=True)
class GateReport:
    results: tuple[GateResult, ...]

    @property
    def blocking(self) -> tuple[GateResult, ...]:
        return tuple(result for result in self.results if result.blocks)

    @property
    def promotable(self) -> bool:
        return not self.blocking

    def describe(self) -> str:
        lines = [result.describe() for result in self.results]
        lines.append("")
        lines.append(
            "promotable"
            if self.promotable
            else f"blocked by {len(self.blocking)} gate(s)"
        )
        return "\n".join(lines)


def _metric(manifest: dict, name: str) -> float | None:
    value = manifest.get("metrics", {}).get(name)
    return float(value) if isinstance(value, (int, float)) else None


def evaluate_gates(
    artifact: Artifact,
    *,
    shadow_period_completed: bool | None = None,
    golden_sets_passed: bool | None = None,
    precision_floor: float | None = None,
    fairness: FairnessReport | None = None,
    reason_reviews: ReviewLedger | None = None,
) -> GateReport:
    """Check an artifact against the promotion gates.

    The keyword arguments are facts this module cannot establish on its own:
    whether a shadow period ran, whether the golden sets passed in CI, what the
    outgoing model's precision floor is, and what the fairness assessment found.
    Left unset they are reported as unevaluated, which blocks — because the
    alternative is a promotion record that claims checks nobody performed.

    ``reason_reviews`` is the exception: the record of who read which reason
    wording ships with the code, so an unset one is read rather than assumed
    absent, and the gate reports a measured figure instead of an unevaluated
    one. It measures 0 accepted today, which blocks, and that is the check
    working rather than a defect in it.
    """
    manifest = artifact.manifest
    results: list[GateResult] = []

    required = {"versions", "splits", "metrics", "threshold", "data", "config"}
    missing = sorted(required - set(manifest))
    results.append(
        GateResult(
            "G1",
            "Manifest complete",
            not missing,
            "all fields present" if not missing else f"missing: {', '.join(missing)}",
        )
    )

    overlap = manifest.get("splits", {}).get("donor_overlap")
    results.append(
        GateResult(
            "G2",
            "No donor overlap",
            overlap == 0 if overlap is not None else None,
            f"donor_overlap={overlap}"
            if overlap is not None
            else "the split summary does not report donor overlap",
        )
    )

    # G3 and G4 are enforced by the test suite on every commit. Their status
    # here reflects whether that suite ran for this artifact, which the
    # artifact cannot know about itself.
    results.append(
        GateResult(
            "G3/G4",
            "Leakage canary and train/serve parity",
            golden_sets_passed,
            "reported by the caller from the CI run"
            if golden_sets_passed is not None
            else "no CI result supplied for this artifact",
        )
    )

    features = manifest.get("versions", {}).get("features")
    results.append(
        GateResult(
            "G5",
            "Metrics attributable",
            bool(features) and bool(manifest.get("trained_at")),
            f"features={features}, trained_at={manifest.get('trained_at')}",
        )
    )

    lift = _metric(manifest, "lift_at_b")
    results.append(
        GateResult(
            "G6",
            "Lift over the rules",
            lift > MIN_LIFT if lift is not None else None,
            f"Lift@B={lift} (must exceed {MIN_LIFT}); at or below it the rules "
            f"surface as much on their own and are cheaper and explainable"
            if lift is not None
            else "no Lift@B recorded",
        )
    )

    precision = _metric(manifest, "precision_at_b")
    if precision_floor is None:
        results.append(
            GateResult(
                "G7",
                "No precision regression",
                None,
                "no incumbent precision floor supplied to compare against",
            )
        )
    else:
        results.append(
            GateResult(
                "G7",
                "No precision regression",
                precision is not None and precision >= precision_floor,
                f"Precision@B={precision} against a floor of {precision_floor}",
            )
        )

    redundancy = manifest.get("redundancy")
    if not isinstance(redundancy, dict):
        results.append(
            GateResult(
                "G8",
                "No redundant features",
                None,
                "the manifest records no redundancy check; an importance "
                "ranking from this run cannot be read, because a duplicated "
                "column splits its importance and neither copy looks important",
            )
        )
    else:
        blocking_kinds = {"identical", "affine"}
        offenders = [
            finding
            for finding in redundancy.get("findings", [])
            if finding.get("kind") in blocking_kinds
        ]
        results.append(
            GateResult(
                "G8",
                "No redundant features",
                None
                if redundancy.get("clean") is None
                else not offenders,
                redundancy.get("unmeasurable_reason")
                or (
                    "; ".join(
                        f"{f['kind']}: {', '.join(f['columns'])}" for f in offenders
                    )
                    if offenders
                    else f"no duplicated column over {redundancy.get('rows')} rows"
                ),
            )
        )

    ece = _metric(manifest, "expected_calibration_error")
    results.append(
        GateResult(
            "G9",
            "Calibration",
            ece <= MAX_CALIBRATION_ERROR if ece is not None else None,
            f"ECE={ece} (must not exceed {MAX_CALIBRATION_ERROR})"
            if ece is not None
            else "no calibration error recorded",
        )
    )

    ledger = reason_reviews if reason_reviews is not None else default_ledger()
    if ledger is None:
        results.append(
            GateResult(
                "G10",
                "Reason wording reviewed",
                None,
                "no record of who reviewed the reason wording is present; the "
                "sentences an analyst triages on cannot be shown to have been "
                "read by anybody",
            )
        )
    else:
        coverage = ledger.coverage()
        results.append(
            GateResult(
                "G10",
                "Reason wording reviewed",
                coverage.complete,
                f"{coverage.describe()}; a reason is what an analyst acts on "
                f"and what a subject is shown, and one nobody has read is not "
                f"an explanation the system can stand behind",
            )
        )

    results.append(
        GateResult(
            "G11",
            "Golden sets",
            golden_sets_passed,
            "reported by the caller from the CI run"
            if golden_sets_passed is not None
            else "no golden-set result supplied for this artifact",
        )
    )

    results.append(
        GateResult(
            "G12",
            "Shadow period completed",
            shadow_period_completed,
            "confirmed by the caller"
            if shadow_period_completed is not None
            else "no shadow period recorded; a model that has not run alongside "
            "the incumbent has not been observed on live traffic",
        )
    )

    if fairness is None:
        results.append(
            GateResult(
                "G13",
                "Fairness assessment",
                None,
                "no fairness assessment supplied; a model whose output tracks "
                "party affiliation looks exactly like one that works, and "
                "nothing else in this report would notice",
            )
        )
    else:
        concerns = fairness.concerns()
        results.append(
            GateResult(
                "G13",
                "Fairness assessment",
                fairness.passed,
                "; ".join(concerns)
                if concerns
                else "no disparity above tolerance on affiliation, district, "
                "recipient type, or size band",
            )
        )

    card = "MODEL_CARD.md"
    results.append(
        GateResult(
            "G14",
            "Model card generated",
            None,
            f"checked against the artifact directory when promoting; {card} must exist",
        )
    )

    return GateReport(tuple(results))


class PromotionRefused(RuntimeError):
    """Raised when a version may not be promoted, with the reason."""


@dataclass(frozen=True)
class Promotion:
    """The record that a person put a model into service."""

    version: str
    approved_by: str
    approved_at: str
    gates: tuple[str, ...]
    note: str | None = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "version": self.version,
                "approved_by": self.approved_by,
                "approved_at": self.approved_at,
                "gates_passed": list(self.gates),
                "note": self.note,
            },
            indent=2,
            sort_keys=True,
        )


def promote(
    version: str,
    *,
    approved_by: str,
    report: GateReport,
    note: str | None = None,
    now: datetime | None = None,
    root: Path = ARTIFACT_ROOT,
) -> Promotion:
    """Record that a version may serve.

    Refuses without an approver and refuses on any blocking gate. There is no
    force argument: a gate that can be waived under pressure is a comment, and
    the pressure is exactly when it matters.
    """
    if not approved_by:
        raise PromotionRefused(
            "a promotion must name the person approving it; the gate exists so "
            "that somebody looked at the numbers and took responsibility"
        )

    directory = root / version
    if not directory.is_dir():
        raise ArtifactError(f"no such version: {version}")

    if not (directory / "MODEL_CARD.md").exists():
        raise PromotionRefused(
            "no model card was generated for this version; a regulator reading "
            "about this system has no other way to learn what it does not detect"
        )

    if report.blocking:
        detail = "\n".join(f"  - {gate.describe()}" for gate in report.blocking)
        raise PromotionRefused(
            f"{len(report.blocking)} gate(s) block promotion:\n{detail}"
        )

    record = Promotion(
        version=version,
        approved_by=approved_by,
        approved_at=(now or datetime.now().astimezone()).isoformat(),
        gates=tuple(result.gate for result in report.results),
        note=note,
    )
    (directory / PROMOTION_FILE).write_text(record.to_json(), encoding="utf-8")
    return record


def promoted_versions(root: Path = ARTIFACT_ROOT) -> tuple[str, ...]:
    if not root.is_dir():
        return ()
    return tuple(
        sorted(
            directory.name
            for directory in root.iterdir()
            if (directory / PROMOTION_FILE).exists()
        )
    )


def current(root: Path = ARTIFACT_ROOT) -> Promotion | None:
    """The version in service, if any.

    Returns None rather than falling back to the newest artifact. Serving an
    unpromoted model because it happened to sort last is precisely what the
    promotion record exists to prevent.
    """
    versions = promoted_versions(root)
    if not versions:
        return None
    data = json.loads((root / versions[-1] / PROMOTION_FILE).read_text(encoding="utf-8"))
    return Promotion(
        version=data["version"],
        approved_by=data["approved_by"],
        approved_at=data["approved_at"],
        gates=tuple(data.get("gates_passed", ())),
        note=data.get("note"),
    )
