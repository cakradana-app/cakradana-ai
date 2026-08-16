"""Command line entry points.

Three commands, each corresponding to a step that used to be a loose script or
a notebook cell:

    generate   build a synthetic dataset and verify it contains its typologies
    train      fit a model, measure it, and say whether it is worth shipping
    score      score a dataset with the current rules and report what fired

The commands share the library the service uses. Nothing here computes a
feature, applies a rule, or decides a threshold on its own — the previous
arrangement, where a notebook held the only implementation of feature
engineering, is what made the shipped model impossible to reproduce or serve.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

from cakradana.data import GeneratorConfig, assert_acceptable, check, generate
from cakradana.features import FeatureService
from cakradana.history import InMemoryDonationStore
from cakradana.rules import RuleEngine, load_latest
from cakradana.training import TrainingConfig, train
from cakradana.training.registry import save


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cakradana", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--seed", type=int, default=GeneratorConfig().seed)
    common.add_argument(
        "--donations",
        type=int,
        default=GeneratorConfig().n_background_donations,
        help="number of ordinary donations to generate",
    )
    common.add_argument(
        "--risky-rate",
        type=float,
        default=GeneratorConfig().risky_rate,
        help="share of donations belonging to a risky pattern",
    )

    generate_cmd = sub.add_parser(
        "generate", parents=[common], help="build and verify a synthetic dataset"
    )
    generate_cmd.add_argument("--out", type=Path, default=None)

    train_cmd = sub.add_parser(
        "train", parents=[common], help="train a model and decide whether it ships"
    )
    train_cmd.add_argument("--budget", type=int, default=100)
    train_cmd.add_argument("--version", default=None, help="artifact version to write")
    train_cmd.add_argument("--artifacts", type=Path, default=Path("artifacts"))

    score_cmd = sub.add_parser(
        "score", parents=[common], help="score a generated dataset with the rules"
    )
    score_cmd.add_argument("--limit", type=int, default=0)

    args = parser.parse_args(argv)
    config = replace(
        GeneratorConfig(),
        seed=args.seed,
        n_background_donations=args.donations,
        risky_rate=args.risky_rate,
    )

    if args.command == "generate":
        return _generate(config, args.out)
    if args.command == "train":
        return _train(config, args)
    return _score(config, args)


def _generate(config: GeneratorConfig, out: Path | None) -> int:
    dataset = generate(config)
    print(json.dumps(dataset.manifest, indent=2))
    print()
    for result in check(dataset):
        print("  " + result.describe())

    try:
        assert_acceptable(dataset)
    except AssertionError as error:
        print(f"\n{error}", file=sys.stderr)
        return 1

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "manifest": dataset.manifest,
                    "donations": [
                        d.model_dump(mode="json") for d in dataset.donations
                    ],
                    "truth": dataset.truth,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {out}")
    return 0


def _context(config: GeneratorConfig):
    dataset = generate(config)
    ruleset = load_latest()
    engine = RuleEngine(
        ruleset,
        calendar=dataset.calendar,
        registers=dataset.registers,
        # Synthetic data, so unverified citations are not being asserted about
        # anyone. Against real records this stays on and an unreviewed
        # statutory rule reports indeterminate instead of a finding.
        require_verified_citations=False,
    )
    features = FeatureService(
        ruleset, calendar=dataset.calendar, registers=dataset.registers
    )
    return dataset, InMemoryDonationStore(dataset.donations), engine, features


def _train(config: GeneratorConfig, args) -> int:
    dataset, store, engine, features = _context(config)
    result = train(
        store,
        engine,
        features,
        truth=dataset.truth,
        entities=dataset.entities,
        config=TrainingConfig(review_budget=args.budget, seed=config.seed),
    )
    print(json.dumps(result.manifest, indent=2))
    print()
    print(result.summary())

    if not result.should_ship:
        print(
            "\nThe rules surface as much as this model adds. They are cheaper, "
            "already explainable, and already running, so the model is not "
            "promoted.",
            file=sys.stderr,
        )

    if args.version:
        directory = save(
            result,
            args.version,
            feature_names=features.names,
            categorical_features=features.categorical,
            root=args.artifacts,
        )
        print(f"\nwrote {directory}")
    return 0


def _score(config: GeneratorConfig, args) -> int:
    from cakradana.scoring.scorer import Scorer

    dataset, store, engine, features = _context(config)
    scorer = Scorer(
        engine.ruleset,
        calendar=dataset.calendar,
        registers=dataset.registers,
        require_verified_citations=False,
    )

    donations = dataset.donations[: args.limit] if args.limit else dataset.donations
    findings: Counter[str] = Counter()
    signals: Counter[str] = Counter()
    indeterminate: Counter[str] = Counter()
    correct = flagged = 0

    for donation in donations:
        result, _ = scorer.score(
            donation,
            store.knowable_at(donation.occurred_at),
            entities=dataset.entities,
        )
        for finding in result.legal_findings:
            findings[finding.rule_id] += 1
        for rule in result.indeterminate_rules:
            indeterminate[rule.rule_id] += 1
        if result.legal_findings:
            flagged += 1
            if donation.donation_id in dataset.truth:
                correct += 1

    print(f"scored {len(donations)} donations\n")
    print("legal findings:")
    for rule_id, count in sorted(findings.items()):
        print(f"  {rule_id}  {count}")
    print(f"\n  of {flagged} donations with a finding, {correct} belong to a "
          f"generated risky pattern")
    print("\nrules that could not be evaluated:")
    for rule_id, count in sorted(indeterminate.items()):
        print(f"  {rule_id}  {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
