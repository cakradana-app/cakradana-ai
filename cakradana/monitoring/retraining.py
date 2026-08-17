"""Deciding when to retrain, and when not to.

Retraining is not automatically an improvement, and the most dangerous case
looks exactly like the healthy one.

Alerts are reviewed; unflagged donations mostly are not. So the labels that
accumulate describe the donations the model already surfaced, and a model
retrained on them learns to agree with its previous self. Precision on that
population climbs, everything reported looks better, and coverage of whatever
the model was already missing quietly narrows. Nothing in the metrics shows it,
because the metrics are computed on the same skewed population.

Two things break the loop, and a retrain that lacks them is refused rather than
run: labels from donations chosen at random rather than because they were
flagged, and a measurement of whether the model finds anything the rules do not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Sequence

from cakradana.schema import Label
from cakradana.schema.enums import HUMAN_LABEL_SOURCES, LabelSource

#: Human-confirmed labels below which a retrain has nothing to learn from.
#: A model refitted on a handful of new judgements mostly reproduces the old
#: one while presenting itself as current.
MIN_HUMAN_LABELS = 200

#: Share of labels that must come from randomly sampled donations rather than
#: from reviewing alerts. Without them, recall cannot be estimated at all: the
#: only confirmed-risky donations anybody has seen are the ones the system
#: already found.
MIN_AUDIT_SHARE = 0.15


@dataclass(frozen=True)
class Trigger:
    """One reason to consider retraining."""

    name: str
    fired: bool
    detail: str

    def describe(self) -> str:
        return f"{'✓' if self.fired else '·'} {self.name}: {self.detail}"


@dataclass
class RetrainingDecision:
    triggers: tuple[Trigger, ...]
    blockers: tuple[str, ...]
    #: Observations that do not block but should reach whoever decides. A
    #: caveat held only in a comment reaches nobody at the moment it matters.
    notes: tuple[str, ...] = ()

    @property
    def any_trigger_fired(self) -> bool:
        return any(t.fired for t in self.triggers)

    @property
    def should_retrain(self) -> bool:
        """Whether to retrain now.

        A blocker overrides every trigger. The conditions that make retraining
        worthwhile and the conditions that make it safe are different
        questions, and drift is a reason to want a new model rather than a
        reason to trust one fitted on unusable labels.
        """
        return self.any_trigger_fired and not self.blockers

    def summary(self) -> str:
        lines = [t.describe() for t in self.triggers]
        if self.notes:
            lines.append("")
            lines.append("worth knowing:")
            lines.extend(f"  - {n}" for n in self.notes)
        if self.blockers:
            lines.append("")
            lines.append("blocked:")
            lines.extend(f"  - {b}" for b in self.blockers)
        lines.append("")
        lines.append(
            "retrain" if self.should_retrain else "do not retrain"
        )
        return "\n".join(lines)


def label_mix(labels: Sequence[Label]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in labels:
        key = str(label.source)
        counts[key] = counts.get(key, 0) + 1
    return counts


def audit_share(labels: Sequence[Label]) -> float:
    """Share of human labels that came from a random sample.

    Identified by note rather than by source, because an analyst's judgement is
    the same kind of evidence whether the donation reached them through the
    queue or through the audit sample. What differs is how the donation was
    chosen, and that is what determines whether recall can be estimated.
    """
    human = [l for l in labels if l.source in HUMAN_LABEL_SOURCES]
    if not human:
        return 0.0
    sampled = sum(1 for l in human if (l.note or "").startswith("audit-sample"))
    return sampled / len(human)


def evaluate(
    labels: Sequence[Label],
    *,
    last_trained_at: datetime | None = None,
    now: datetime | None = None,
    drift_detected: bool = False,
    pipeline_faults: Sequence[str] = (),
    rule_set_changed: bool = False,
    max_model_age: timedelta = timedelta(days=90),
) -> RetrainingDecision:
    """Decide whether a retrain is warranted and whether it is safe."""
    now = now or datetime.now().astimezone()
    human = [l for l in labels if l.source in HUMAN_LABEL_SOURCES]
    sampled_share = audit_share(labels)

    triggers = (
        Trigger(
            name="new human judgements",
            fired=len(human) >= MIN_HUMAN_LABELS,
            detail=f"{len(human)} confirmed labels (threshold {MIN_HUMAN_LABELS})",
        ),
        Trigger(
            name="input or score drift",
            fired=drift_detected,
            detail="the population has moved away from what the model was fitted to"
            if drift_detected
            else "no material drift",
        ),
        Trigger(
            name="rule set changed",
            fired=rule_set_changed,
            detail="the heuristics that produce training labels have changed, so the "
            "model's baseline has moved"
            if rule_set_changed
            else "rules unchanged",
        ),
        Trigger(
            name="model age",
            fired=bool(last_trained_at and now - last_trained_at > max_model_age),
            detail=(
                f"last trained {(now - last_trained_at).days} days ago"
                if last_trained_at
                else "never trained"
            ),
        ),
    )

    blockers: list[str] = []

    if len(human) < MIN_HUMAN_LABELS:
        blockers.append(
            f"only {len(human)} human-confirmed labels are available; a model "
            f"refitted on this many mostly reproduces the previous one while "
            f"presenting itself as current"
        )

    if sampled_share < MIN_AUDIT_SHARE:
        blockers.append(
            f"only {sampled_share:.0%} of confirmed labels come from randomly "
            f"sampled donations (need {MIN_AUDIT_SHARE:.0%}). Labels drawn only "
            f"from reviewed alerts describe the donations the model already "
            f"found, so retraining on them teaches it to agree with itself "
            f"while its coverage of everything else narrows unmeasured"
        )

    if pipeline_faults:
        blockers.append(
            "features stopped being computable ("
            + ", ".join(pipeline_faults)
            + "); retraining now would fit the fault rather than the data"
        )

    notes: list[str] = []
    if not any(l.source is LabelSource.DISPUTE_OUTCOME for l in labels) and len(human) >= MIN_HUMAN_LABELS:
        # Not a blocker: a corpus of analyst judgements is usable on its own.
        # But adjudicated outcomes are the strongest evidence this system can
        # hold, and a label set containing none of them was assembled without
        # any contested attribution ever being resolved — which is worth
        # knowing before fitting a model to it, and reaches nobody if it lives
        # in a comment.
        notes.append(
            "no adjudicated dispute outcomes among the human labels; the "
            "strongest evidence available is absent from the training set"
        )

    return RetrainingDecision(
        triggers=triggers, blockers=tuple(blockers), notes=tuple(notes)
    )
