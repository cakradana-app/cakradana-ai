"""The reason-code catalogue and its wording rules.

Two failures are guarded against here, and they are opposite ones.

A code emitted but not catalogued means the catalogue is incomplete while
reading as a complete enumeration, which is worse than having none: the review
state it reports would cover only the codes somebody remembered to add. So the
catalogue is checked against what the source actually constructs and against
what a generated dataset actually emits.

A code catalogued but worded as a conclusion means the system has decided the
case in the sentence the analyst reads first. The wording rules that catch that
are machine-checkable and are checked, which is the floor an analyst's reading
starts from — not a substitute for it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import cakradana
from cakradana.data import GeneratorConfig, generate
from cakradana.features.definitions import feature_names
from cakradana.history import InMemoryDonationStore
from cakradana.lanes.alerts import AlertIndex, AlertKind, DetectorSettings, GroupAlertDetector
from cakradana.lanes.graph import STRUCTURAL_RULES
from cakradana.rules import load_latest
from cakradana.scoring.catalogue import (
    ReasonCode,
    catalogue,
    codes,
    entry_for,
    wording_defects,
)
from cakradana.scoring.composition import ScoreComposer
from cakradana.scoring.result import Lane, Reason, ReviewStatus
from cakradana.scoring.review import default_statuses
from cakradana.scoring.scorer import Scorer
from tests.conftest import make_donation

PACKAGE = Path(cakradana.__file__).resolve().parent

#: Modules allowed to build a reason code from a value rather than a literal.
#: Each names a domain that is enumerated in full below, so a dynamic code is
#: still a code the catalogue declares. A new module constructing one fails,
#: because nothing here would know what its codes are.
DERIVED_IN = {
    "lanes/graph.py": "the structural rule ids and the group alert kinds",
    "lanes/classifier.py": "the feature set",
}


def undeclared(emitted) -> tuple[str, ...]:
    """Codes that were emitted and are not in the catalogue."""
    return tuple(sorted(set(emitted) - set(codes())))


def _reason_calls():
    """Every ``Reason(...)`` construction in the package, by source file.

    Parsed rather than imported. A construction site inside a branch that no
    test reaches is still a sentence the system can show somebody.
    """
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (
                func.id
                if isinstance(func, ast.Name)
                else func.attr if isinstance(func, ast.Attribute) else None
            )
            if name == "Reason":
                yield path.relative_to(PACKAGE).as_posix(), node


def _code_argument(node: ast.Call):
    for keyword in node.keywords:
        if keyword.arg == "code":
            return keyword.value
    return None


@pytest.fixture(scope="module")
def scored():
    """Codes and statements a generated population actually produces.

    The graph lane is given detected clusters, because the alert wordings are
    only reachable through them and an unreachable wording is exactly the kind
    the catalogue would be wrong about.
    """
    dataset = generate(GeneratorConfig(seed=7, n_background_donations=400))
    store = InMemoryDonationStore(dataset.donations)
    as_of = max(d.occurred_at for d in dataset.donations)
    detector = GroupAlertDetector(DetectorSettings())
    alerts = AlertIndex(detector.detect(store.knowable_at(as_of), as_of=as_of))

    scorer = Scorer(
        load_latest(),
        calendar=dataset.calendar,
        registers=dataset.registers,
        alerts=alerts,
        require_verified_citations=False,
    )

    emitted: list[tuple[str, Lane, str]] = []
    for donation in dataset.donations:
        result, _ = scorer.score(
            donation,
            store.knowable_at(donation.occurred_at),
            entities=dataset.entities,
        )
        if result.behavioural is None:
            continue
        for reason in result.behavioural.reasons:
            emitted.append((reason.code, reason.lane, reason.statement))
    return tuple(emitted)


def _compose(statuses: dict[str, ReviewStatus]):
    """Score one donation with an explicit review state per code.

    Three lanes are not configured, so the result always carries the composer's
    own reason for their absence — the one code every result contains, and the
    one this can assert on without depending on a detection firing.
    """
    donation = make_donation()
    store = InMemoryDonationStore([donation])
    scorer = Scorer(
        load_latest(),
        require_verified_citations=False,
        composer=ScoreComposer(wording_statuses=statuses),
    )
    return scorer.score(donation, store.knowable_at(donation.occurred_at))


def _all(status: ReviewStatus) -> dict[str, ReviewStatus]:
    return {code: status for code in codes()}


@pytest.fixture
def composed():
    return _compose({})


class TestTheCatalogueEnumeratesEverything:
    def test_every_literal_code_in_the_source_is_declared(self):
        literals = {
            node_code.value
            for _, node in _reason_calls()
            if isinstance(node_code := _code_argument(node), ast.Constant)
            and isinstance(node_code.value, str)
        }
        assert literals, "no reason construction sites were found to check"
        assert not undeclared(literals)

    def test_a_code_built_from_a_value_comes_from_an_enumerated_domain(self):
        """A dynamic code is fine; a dynamic code nothing enumerates is not."""
        dynamic = {
            module
            for module, node in _reason_calls()
            if not isinstance(_code_argument(node), ast.Constant)
        }
        assert dynamic, "no dynamically coded reasons were found to check"
        assert not dynamic - set(DERIVED_IN)

    def test_every_group_alert_kind_is_declared(self):
        assert not undeclared(str(kind) for kind in AlertKind)

    def test_every_structural_rule_code_is_declared(self):
        assert not undeclared(STRUCTURAL_RULES.values())

    def test_every_feature_the_classifier_can_name_is_declared(self):
        assert not undeclared(name.upper() for name in feature_names())

    def test_every_code_emitted_over_a_generated_dataset_is_declared(self, scored):
        assert scored, "the generated population produced no reasons at all"
        assert not undeclared(code for code, _, _ in scored)

    def test_every_emitted_lane_is_declared_for_its_code(self, scored):
        wrong = sorted(
            {
                (code, str(lane))
                for code, lane, _ in scored
                if lane not in entry_for(code).lanes
            }
        )
        assert not wrong

    def test_emitted_wording_still_matches_the_catalogued_template(self, scored):
        """Catches the wording drifting away from the wording under review.

        A statement edited in a lane while the catalogue keeps the old text
        means an analyst accepted a sentence the system no longer says.
        """
        drifted = sorted(
            {
                (code, statement)
                for code, _, statement in scored
                if not entry_for(code).matches(statement)
            }
        )
        assert not drifted

    def test_an_uncatalogued_code_is_reported_rather_than_ignored(self):
        """The check above only means something if it can fail."""
        assert undeclared(["FAN_IN_BURST", "INVENTED_CODE"]) == ("INVENTED_CODE",)

    def test_no_code_is_declared_twice(self):
        assert len(set(codes())) == len(codes())


class TestWordingStatesAnObservation:
    def test_no_catalogued_wording_carries_a_defect(self):
        defective = {
            entry.code: wording_defects(entry)
            for entry in catalogue()
            if wording_defects(entry)
        }
        assert not defective

    def test_every_code_names_what_produced_it(self):
        assert all(entry.lanes and entry.source.strip() for entry in catalogue())

    def test_every_code_carries_the_wording_it_is_reviewed_on(self):
        assert all(
            entry.statements and all(s.strip() for s in entry.statements)
            for entry in catalogue()
        )

    @pytest.mark.parametrize(
        "statement",
        [
            "This donor is suspicious and the donation is likely fraud.",
            "The pattern indicates smurfing across nine days.",
            "The donation therefore violates the limit.",
        ],
    )
    def test_a_conclusion_is_a_defect(self, statement):
        assert wording_defects(_entry(statement))

    @pytest.mark.parametrize(
        "statement",
        [
            "Feature importance 0.42 pushed the score up.",
            "The logit for this donation was 2.1.",
            "A z-score of 3.4 was observed.",
        ],
    )
    def test_a_model_internal_is_a_defect(self, statement):
        assert wording_defects(_entry(statement))

    @pytest.mark.parametrize(
        "statement",
        [
            "This donation scored 82 and was flagged.",
            "The probability of risk is 0.7.",
        ],
    )
    def test_restating_the_score_is_a_defect(self, statement):
        assert wording_defects(_entry(statement))

    def test_an_observation_is_not_a_defect(self):
        assert not wording_defects(
            _entry("23 distinct senders donated to PARTAI X within 9 days.")
        )

    def test_a_code_naming_no_source_is_a_defect(self):
        entry = _entry("23 distinct senders donated within 9 days.").model_copy(
            update={"source": "  "}
        )
        assert wording_defects(entry)


def _entry(statement: str) -> ReasonCode:
    return ReasonCode(
        code="EXAMPLE",
        lanes=(Lane.GRAPH,),
        source="RULE-T2-01",
        observation="an example",
        statements=(statement,),
    )


class TestTheResultSaysWhetherAnybodyReadIt:
    """An unreviewed wording must not render the same as an accepted one.

    An analyst reading a case bundle has no other way to tell that the sentence
    in front of them has never been vetted, and a system that reads the same
    either way spends the reviewing they did do on the codes they did not.
    """

    def test_a_reason_built_anywhere_defaults_to_unreviewed(self):
        reason = Reason(
            code="FAN_IN_BURST", lane=Lane.GRAPH, weight=0.5, statement="x"
        )
        assert reason.wording_review is ReviewStatus.UNREVIEWED
        assert not reason.wording_review.is_acceptable

    def test_every_reason_in_a_scored_result_carries_a_review_state(self, composed):
        result, _ = composed
        assert result.behavioural.reasons
        assert all(
            isinstance(r.wording_review, ReviewStatus)
            for r in result.behavioural.reasons
        )

    def test_the_state_travels_into_the_serialised_result(self, composed):
        result, _ = composed
        payload = result.model_dump(mode="json")["behavioural"]
        assert {r["wording_review"] for r in payload["reasons"]} == {"unreviewed"}
        assert set(payload["unreviewed_wording"]) == {
            r["code"] for r in payload["reasons"]
        }

    def test_an_accepted_wording_does_not_render_as_an_unreviewed_one(self):
        unreviewed, _ = _compose({})
        accepted, _ = _compose(_all(ReviewStatus.VALIDATED))
        assert {r.wording_review for r in unreviewed.behavioural.reasons} == {
            ReviewStatus.UNREVIEWED
        }
        assert {r.wording_review for r in accepted.behavioural.reasons} == {
            ReviewStatus.VALIDATED
        }
        assert accepted.behavioural.unreviewed_wording == ()
        assert unreviewed.model_dump() != accepted.model_dump()

    def test_a_rejected_wording_is_reported_apart_from_an_unread_one(self):
        """Somebody looked and said no. That is a different problem from
        nobody having looked, and collapsing them loses the one that has an
        owner."""
        statuses = {**_all(ReviewStatus.VALIDATED)}
        statuses["LANE_UNAVAILABLE"] = ReviewStatus.REJECTED
        result, _ = _compose(statuses)
        assert result.behavioural.rejected_wording == ("LANE_UNAVAILABLE",)
        assert result.behavioural.unreviewed_wording == ()

    def test_the_lane_and_the_summary_agree(self, composed):
        result, _ = composed
        within_lanes = {
            r.wording_review
            for lane in result.behavioural.lanes
            for r in lane.reasons
        }
        assert within_lanes <= {ReviewStatus.UNREVIEWED}

    def test_the_shipped_ledger_leaves_every_emitted_code_unreviewed(self, scored):
        assert {code for code, _, _ in scored}
        assert all(
            default_statuses().get(code, ReviewStatus.UNREVIEWED)
            is ReviewStatus.UNREVIEWED
            for code, _, _ in scored
        )
