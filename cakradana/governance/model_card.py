"""Model cards, generated from the run rather than written about it.

A card composed by hand describes what somebody believed at the time. This one
is assembled from the training manifest, so it cannot claim a figure the run did
not produce, and it states the limits alongside the numbers instead of leaving
them to a reader's judgement.

That matters here because of what went before. The performance figures
published for this system came from a different model than the one that
shipped, were inflated by aggregates computed across the whole dataset before
it was split, and were reported as accuracy on a set constructed to be half
risky. Every one of those is invisible in a number and obvious in a card that
has to say where the number came from.
"""

from __future__ import annotations

from datetime import datetime
from typing import Mapping

INTENDED_USE = """\
Prioritising political donation records for human investigation.

The output ranks donations for attention. It does not determine that an offence
occurred, and no part of it should be presented as such."""

OUT_OF_SCOPE = """\
- Deciding any matter without human review.
- Scoring people. The unit of analysis is a donation and its network position,
  not a person's character.
- Establishing a legal violation. Statutory findings come from the rule engine
  with a citation; this model produces an estimate about conduct.
- Any use where a false positive is acted on directly. False positives here
  concern named individuals in a political context."""


def _fmt(value: object, digits: int = 3) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def generate(
    manifest: Mapping[str, object],
    *,
    model_version: str,
    generated_at: datetime | None = None,
) -> str:
    """Render a model card from a training manifest."""
    metrics = manifest.get("metrics", {}) or {}
    data = manifest.get("data", {}) or {}
    versions = manifest.get("versions", {}) or {}
    splits = manifest.get("splits", {}) or {}
    label_basis = manifest.get("label_basis", {}) or {}
    config = manifest.get("config", {}) or {}

    human_labels = bool(label_basis.get("is_human_confirmed"))
    ships = bool(manifest.get("ships"))
    lift = metrics.get("lift_at_b")

    stamp = (generated_at or datetime.now().astimezone()).isoformat(timespec="seconds")

    lines: list[str] = [
        f"# Model card — {model_version}",
        "",
        f"Generated {stamp} from the training run that produced this artifact.",
        "",
        "## Status",
        "",
    ]

    if not ships:
        lines += [
            "**This model is not promoted.**",
            "",
            f"Its incremental yield over the rule engine is {_fmt(lift, 2)}. At or "
            "below parity the rules surface as much as the model adds, and they "
            "are cheaper to run, already explainable, and already deployed. The "
            "rules operate alone.",
            "",
        ]
    else:
        lines += [
            "**Promoted.**",
            "",
            f"Incremental yield over the rule engine is {_fmt(lift, 2)}: it "
            "surfaces confirmed-risky donations that no rule flagged.",
            "",
        ]

    if not human_labels:
        lines += [
            "> **The figures below are not measurements of detection performance.**",
            ">",
            "> They were computed against "
            f"{label_basis.get('source', 'generated')} labels, which say whether "
            "the model recovered patterns that were planted for it to find. "
            "That is a different claim from finding real risk, and reporting one "
            "as the other is how a demonstration becomes a claim.",
            "",
        ]

    lines += [
        "## Intended use",
        "",
        INTENDED_USE,
        "",
        "## Out of scope",
        "",
        OUT_OF_SCOPE,
        "",
        "## How it was measured",
        "",
        f"- Review budget: top {metrics.get('budget', '—')} donations by score.",
        f"- Precision@B: {_fmt(metrics.get('precision_at_b'))}",
        f"- Recall@B: {_fmt(metrics.get('recall_at_b'))}",
        f"- Lift@B: {_fmt(lift, 2)} "
        f"({metrics.get('novel_finds', 0)} confirmed finds no rule flagged, "
        f"against {metrics.get('rule_baseline_finds', 0)} the rules surface alone)",
        f"- Average precision: {_fmt(metrics.get('average_precision'))}",
        f"- Expected calibration error: {_fmt(metrics.get('expected_calibration_error'), 4)}",
        f"- Operating threshold: {_fmt(manifest.get('threshold'), 4)}",
        "",
        "Accuracy and F1 are deliberately absent. On a population where a few "
        "percent of donations are risky, a model that flags nothing scores above "
        "95% accuracy, so reporting it would be worse than reporting nothing.",
        "",
        "Recall here is bounded by what has been reviewed. A recall figure is "
        "only an estimate of true recall once unflagged donations are sampled "
        "and reviewed too; without that, the denominator counts only cases that "
        "were confirmed because the system surfaced them.",
        "",
        "## Data",
        "",
        f"- Training rows: {data.get('rows', '—')}",
        f"- Positive rate: {_fmt(data.get('base_rate'), 4)} "
        "(the real class distribution, not resampled to look balanced)",
        f"- Class weighting: {_fmt(data.get('scale_pos_weight'), 2)}",
        f"- Label basis: {label_basis.get('source', '—')}",
        "",
        "Training labels come from behavioural heuristics at reduced weight. "
        "They are hypotheses about intent inferred from structure, not "
        "observations. Statutory outcomes are deliberately excluded as targets: "
        "a model trained on those could only relearn arithmetic it was already "
        "given.",
        "",
        "## Splits",
        "",
    ]

    for name in ("train", "calibration", "test"):
        split = splits.get(name, {}) or {}
        lines.append(
            f"- {name}: {split.get('donations', '—')} donations, "
            f"{split.get('donors', '—')} donors"
        )

    lines += [
        "",
        "Splits are grouped by donor and the absence of overlap is asserted, not "
        "reported. A donor on both sides lets the model recognise them instead of "
        "generalising, and donor behaviour is most of what these features "
        "describe.",
        "",
        "Donors are assigned to cohorts by first appearance, so what is measured "
        "is generalisation to donors never seen before. It does not measure "
        "forecasting a later period from an earlier one.",
        "",
        "## Versions",
        "",
        f"- Model: {model_version}",
        f"- Rule set: {versions.get('rule_set', '—')}",
        f"- Feature definitions: {versions.get('features', '—')}",
        f"- Trained at: {manifest.get('trained_at', '—')}",
        f"- Seed: {config.get('seed', '—')}",
        "",
        "## Known limitations",
        "",
        _limitations(human_labels, ships),
        "",
        "## What is not evaluated at all",
        "",
        "Several statutory prohibitions depend on reference data that does not "
        "exist yet — donor jurisdiction, an authoritative register of state "
        "bodies and enterprises, a register of convictions with final legal "
        "force, and campaign finance submissions for reconciliation. Those rules "
        "report indeterminate rather than passing, and their absence from a "
        "donation's findings is not evidence of compliance.",
    ]

    return "\n".join(lines) + "\n"


def _limitations(human_labels: bool, ships: bool) -> str:
    items = [
        "- The model ranks; it does not determine. Every surfaced donation "
        "requires human review before any action.",
        "- Cumulative limit findings depend entirely on entity resolution. A "
        "donor split across unresolved name variants evades them, which is the "
        "behaviour those limits exist to catch.",
        "- Scores are comparable only within a model version. Where score "
        "semantics change, the bands are re-tuned and the change is announced.",
    ]
    if not human_labels:
        items.append(
            "- No performance figure here describes real detection. Until "
            "human-confirmed labels exist, every number states how well the "
            "model recovered planted patterns."
        )
    if not ships:
        items.append(
            "- This artifact did not meet the bar for adding detection over the "
            "rule engine and is not in service."
        )
    return "\n".join(items)


def write(
    manifest: Mapping[str, object],
    path,
    *,
    model_version: str,
    generated_at: datetime | None = None,
) -> None:
    from pathlib import Path

    Path(path).write_text(
        generate(manifest, model_version=model_version, generated_at=generated_at),
        encoding="utf-8",
    )
