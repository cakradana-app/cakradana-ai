"""Command line entry points.

Five commands, each corresponding to a step that used to be a loose script, a
notebook cell, or nothing at all:

    generate      build a synthetic dataset and verify it contains its typologies
    train         fit a model, measure it, and say whether it is worth shipping
    score         score a dataset with the current rules and report what fired
    benchmark     measure scoring latency, and whether cost grows with history
    reason-codes  read the wording shown to analysts, and record a decision on it

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
from cakradana.evaluation.timing import ScalingReport, measure
from cakradana.features import FeatureService
from cakradana.history import InMemoryDonationStore
from cakradana.rules import RuleEngine, load_latest
from cakradana.scoring.catalogue import catalogue, entry_for, wording_defects
from cakradana.scoring.result import ReviewStatus
from cakradana.scoring.review import (
    REVIEW_FILE,
    ReviewDecision,
    ReviewLedger,
    ReviewRefused,
    default_ledger,
    now,
)
from cakradana.training import TrainingConfig, train
from cakradana.serving.service import ScoringService
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

    benchmark_cmd = sub.add_parser(
        "benchmark",
        parents=[common],
        help="measure scoring latency and how it scales with history",
    )
    benchmark_cmd.add_argument("--samples", type=int, default=60)
    benchmark_cmd.add_argument(
        "--factor",
        type=int,
        default=4,
        help="how much larger the second population is, for the scaling reading",
    )

    reasons_cmd = sub.add_parser(
        "reason-codes",
        help="read the wording analysts are shown, and record a decision on it",
    )
    _add_reason_code_actions(reasons_cmd)

    args = parser.parse_args(argv)

    # Dispatched before the generator config is built: reviewing wording has
    # nothing to do with synthetic data, and the shared options would only
    # invite the reader to think it did.
    if args.command == "reason-codes":
        return _reason_codes(args)

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
    if args.command == "benchmark":
        return _benchmark(config, args)
    return _score(config, args)


def _add_reason_code_actions(command: argparse.ArgumentParser) -> None:
    """Wire the four things an analyst does with reason wording.

    There is no bulk accept. Reviewing a wording means reading the sentence,
    and a switch that accepted fifty at once would produce a ledger certifying
    nothing while reading as complete.
    """
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--file",
        type=Path,
        default=REVIEW_FILE,
        help="ledger to read and write (defaults to the one beside the code)",
    )

    actions = command.add_subparsers(dest="reason_action", required=True)

    listing = actions.add_parser(
        "list", parents=[shared], help="every code, with its review state"
    )
    listing.add_argument(
        "--status",
        choices=("unreviewed", "validated", "rejected", "all"),
        default="unreviewed",
        help="which codes to show (default: the ones nobody has read)",
    )

    show = actions.add_parser(
        "show", parents=[shared], help="one code in full, as it will be read"
    )
    show.add_argument("code")

    for name, help_text in (
        ("accept", "record that the wording reads as an observation"),
        ("reject", "record that the wording is misleading, and why"),
    ):
        decision = actions.add_parser(name, parents=[shared], help=help_text)
        decision.add_argument("code")
        decision.add_argument(
            "--reviewer",
            required=True,
            help="who read it; a decision nobody is answerable for is not a review",
        )
        decision.add_argument(
            "--note",
            required=True,
            help="what was checked, or what was wrong with it",
        )

    actions.add_parser(
        "coverage",
        parents=[shared],
        help="how much of the catalogue anybody has read (non-zero when short)",
    )


def _reason_codes(args) -> int:
    if args.reason_action == "list":
        return _list_reason_codes(args)
    if args.reason_action == "show":
        return _show_reason_code(args)
    if args.reason_action == "coverage":
        return _reason_code_coverage(args)
    return _record_reason_code_decision(args)


def _ledger(path: Path) -> ReviewLedger:
    loaded = ReviewLedger.load(path)
    return loaded if loaded is not None else ReviewLedger()


def _list_reason_codes(args) -> int:
    """List the codes with the sentence each one puts in front of a reader.

    The wording is what is under review, so the wording is what is printed. A
    listing that showed only a code and a summary would be a list of labels,
    and a reviewer would have to open every one to find out what the system
    actually says.
    """
    ledger = _ledger(args.file)
    barred = [entry for entry in catalogue() if not entry.analyst_facing]
    shown = 0
    for entry in catalogue():
        if not entry.analyst_facing:
            continue
        status = ledger.status_of(entry.code)
        if args.status != "all" and status.value != args.status:
            continue
        shown += 1
        print(f"  {entry.code}  [{status}]")
        for statement in entry.statements:
            print(f"      {statement}")
    if not shown:
        print(f"  no code is {args.status}")

    if barred:
        print(
            f"\n{len(barred)} further code(s) carry no wording and are never "
            f"shown to anybody, so there is nothing to review:"
        )
        for entry in barred:
            print(f"  {entry.code}  — {entry.observation}")

    print(f"\n{ledger.coverage().describe()}")
    return 0


def _show_reason_code(args) -> int:
    entry = entry_for(args.code)
    if entry is None:
        print(
            f"{args.code} is not a code this system emits; `reason-codes list "
            f"--status all` shows the ones it does",
            file=sys.stderr,
        )
        return 1

    ledger = _ledger(args.file)
    print(f"{entry.code}")
    print(f"  lane      {'/'.join(str(lane) for lane in entry.lanes)}")
    print(f"  from      {entry.source}")
    print(f"  states    {entry.observation}")
    print(f"  review    {ledger.status_of(entry.code)}")
    decision = ledger.decision_for(entry.code)
    if decision:
        print(f"            by {decision.reviewer} on {decision.reviewed_at.isoformat()}")
        print(f"            {decision.note}")
    print("\n  wording as an analyst reads it:")
    for statement in entry.statements:
        print(f"    {statement}")
    defects = wording_defects(entry)
    if defects:
        print("\n  machine-checkable problems:")
        for defect in defects:
            print(f"    {defect}")
    return 0


def _record_reason_code_decision(args) -> int:
    """Write one decision, or refuse and say why.

    The reviewer and the note are required by the parser rather than prompted
    for, so a decision cannot be recorded by pressing return past a question.
    """
    try:
        entry = entry_for(args.code)
        if entry is None:
            raise ReviewRefused(
                f"{args.code} is not a code this system emits; a review of "
                f"wording nobody will ever see is not coverage of anything"
            )
        if not entry.analyst_facing:
            raise ReviewRefused(
                f"{args.code} carries no wording and is never shown to "
                f"anybody, so there is nothing to accept or reject: "
                f"{entry.observation}"
            )
        decision = ReviewDecision(
            code=args.code,
            status=(
                ReviewStatus.VALIDATED
                if args.reason_action == "accept"
                else ReviewStatus.REJECTED
            ),
            reviewer=args.reviewer,
            reviewed_at=now(),
            note=args.note,
            # The sentence as it stands now, so a later amendment to it puts
            # the code back in the queue instead of inheriting this decision.
            statements=entry.statements,
        )
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1

    ledger = _ledger(args.file).record(decision)
    ledger.save(args.file)
    # The next read in this process must see what was just written.
    default_ledger.cache_clear()

    print(f"{decision.code} recorded as {decision.status} by {decision.reviewer}")
    print(f"wrote {args.file}")
    print(f"\n{ledger.coverage().describe()}")
    print(
        "\nCommit the ledger. A review that exists only on the machine it was "
        "taken on is not a record anybody else can read."
    )
    return 0


def _reason_code_coverage(args) -> int:
    coverage = _ledger(args.file).coverage()
    print(coverage.describe())
    if coverage.complete is True:
        return 0
    print(
        "\nA reason is what an analyst acts on and what a subject is shown. "
        "One nobody has read is not an explanation this system can stand "
        "behind, and promotion gate G10 blocks on it.",
        file=sys.stderr,
    )
    return 1


def _benchmark(config: GeneratorConfig, args) -> int:
    """Take a latency reading on this machine.

    The wall-clock figures belong to whatever runs them, which is exactly why
    this exists as a command rather than as a number in a document: the target
    is only meaningful against a reading somebody actually took, on the hardware
    the system is going to run on.

    The scaling ratio is the part that travels. It is a property of the code,
    so a sub-linear result here means the same thing anywhere.
    """
    small = _timed(config, args.donations, args.samples)
    large = _timed(config, args.donations * args.factor, args.samples)
    report = ScalingReport("score", small, large)
    print(report.describe())
    if report.is_sublinear is None:
        print("\nscaling could not be measured", file=sys.stderr)
        return 1
    if not report.is_sublinear:
        # Reported as a failure rather than a note. A system whose cost tracks
        # its history gets slower as it succeeds, and production is where that
        # is otherwise discovered.
        print("\nscoring cost grows with history", file=sys.stderr)
        return 1
    return 0


def _timed(config: GeneratorConfig, donations: int, samples: int):
    # The donor pool grows with the population. Holding it fixed would make a
    # larger dataset mean the same donors giving more often, which is a
    # different shape of history and not the one being scaled.
    scale = max(donations / max(config.n_background_donations, 1), 1.0)
    dataset, store, engine, features = _context(
        replace(
            config,
            n_background_donations=donations,
            n_legitimate_donors=max(20, int(config.n_legitimate_donors * scale)),
        )
    )
    service = ScoringService(
        calendar=dataset.calendar,
        registers=dataset.registers,
        entities=dataset.entities,
        require_verified_citations=False,
    )
    service.replay(dataset.donations, entities=dataset.entities)

    def call(index: int) -> object:
        donation = dataset.donations[index % len(dataset.donations)]
        # Not remembered: the history each call is judged against has to stay
        # the size being measured rather than growing under the measurement.
        view = service.store.knowable_at(donation.occurred_at)
        return service.scorer.score(donation, view, entities=service.entities)

    return measure("score", call, samples=samples, population=len(service.store))


def _generate(config: GeneratorConfig, out: Path | None) -> int:
    dataset = generate(config)
    print(json.dumps(dataset.manifest, indent=2))
    print()
    print(dataset.retirement_notice())
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
    if dataset.is_retired():
        # Refused rather than warned about. A model trained on data that has
        # outlived the rule set it was generated under carries metrics nobody
        # can attribute, and the metrics are what get quoted.
        print(f"\n{dataset.retirement_notice()}", file=sys.stderr)
        return 1
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
