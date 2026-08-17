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
from datetime import timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

import cakradana
from cakradana.data import GeneratorConfig, generate
from cakradana.features import FeatureService, FeatureVector
from cakradana.features.definitions import feature_names
from cakradana.history import InMemoryDonationStore
from cakradana.lanes.alerts import AlertIndex, AlertKind, DetectorSettings, GroupAlertDetector
from cakradana.lanes.anomaly import AnomalyLane, fit
from cakradana.lanes.classifier import ClassifierLane
from cakradana.lanes.graph import STRUCTURAL_RULES
from cakradana.lanes.reputation import (
    CoverageIndex,
    CoverageItem,
    OperatingConditions,
    ReputationLane,
)
from cakradana.rules import load_latest
from cakradana.scoring.catalogue import (
    _LABEL_AND_VALUE,
    ReasonCode,
    catalogue,
    codes,
    entry_for,
    wording_defects,
)
from cakradana.scoring.composition import ScoreComposer
from cakradana.scoring.result import Lane, Reason, ReviewStatus
from cakradana.scoring.review import default_statuses
from cakradana.scoring.scorer import GraphLaneAdapter, Scorer
from cakradana.training.registry import Artifact
from tests.conftest import at, make_donation

PACKAGE = Path(cakradana.__file__).resolve().parent

#: Modules allowed to build a reason code from a value rather than a literal.
#: Each names a domain that is enumerated in full below, so a dynamic code is
#: still a code the catalogue declares. A new module constructing one fails,
#: because nothing here would know what its codes are.
DERIVED_IN = {
    "lanes/graph.py": "the structural rule ids and the group alert kinds",
    "lanes/classifier.py": "the feature set",
}


#: Wordings the generated population cannot produce, and why. Named rather
#: than left as a shortfall in a count: each is a sentence the system can show
#: somebody that nothing here checks, and the way that stops being true is a
#: population that contains the shape, not a looser assertion.
UNREACHED_BY_THIS_POPULATION = {
    "FAN_OUT": (
        "no donor in the generated population reaches enough distinct "
        "recipients inside the window to fire the rule or the alert"
    ),
    "LAYERING_CHAIN": (
        "the generator builds no chain passing through intermediate parties"
    ),
    "HAS_UNRESOLVED_ENTITY": (
        "every party in the generated population resolves, so the flag is "
        "false throughout and a false boolean is correctly not a reason"
    ),
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

    Two passes, because no single one reaches every wording. The population is
    scored with the graph lane holding detected clusters, since the alert
    wordings are only reachable through them. Then every statement the
    classifier can make is drawn from real feature values, since no model ships
    and the lane would otherwise contribute nothing at all — leaving forty-odd
    catalogued sentences checked by nothing.
    """
    dataset = generate(GeneratorConfig(seed=7, n_background_donations=400))
    store = InMemoryDonationStore(dataset.donations)
    as_of = max(d.occurred_at for d in dataset.donations)
    detector = GroupAlertDetector(DetectorSettings())
    alerts = AlertIndex(detector.detect(store.knowable_at(as_of), as_of=as_of))

    def pass_over(lanes):
        scorer = Scorer(
            load_latest(),
            calendar=dataset.calendar,
            registers=dataset.registers,
            alerts=alerts,
            lanes=lanes,
            require_verified_citations=False,
        )
        found: list[tuple[str, Lane, str]] = []
        vectors: list[FeatureVector] = []
        for donation in dataset.donations:
            result, features = scorer.score(
                donation,
                store.knowable_at(donation.occurred_at),
                entities=dataset.entities,
            )
            vectors.append(features)
            if result.behavioural is None:
                continue
            for reason in result.behavioural.reasons:
                found.append((reason.code, reason.lane, reason.statement))
        return found, vectors

    graph = GraphLaneAdapter(alerts)
    _, vectors = pass_over([graph])

    # The exploratory lanes are switched off in this system, so a pass that
    # only ran what is configured would leave their wordings checked by
    # nothing. Fitted and supplied here to make them speak.
    emitted, _ = pass_over(
        [
            graph,
            AnomalyLane(fit(vectors, FeatureService(load_latest()))),
            ReputationLane(_coverage_for(dataset), _ALL_CONDITIONS_MET),
        ]
    )
    emitted.extend(_classifier_emissions(vectors))
    return tuple(emitted)


#: Every precondition the reputation lane refuses to run without. Set here so
#: the lane will speak; in the system they are all unmet and it does not.
_ALL_CONDITIONS_MET = OperatingConditions(
    defamation_review_completed=True,
    source_list_published=True,
    matching_accuracy_measured=True,
    subject_access_route_exists=True,
    retraction_handling_implemented=True,
    named_owner="compliance@example.org",
    lift_measured=True,
)


def _coverage_for(dataset) -> CoverageIndex:
    """Adverse coverage about one donor the population actually contains.

    Published before the donation it has to reach. Coverage that appeared after
    a donation could not have been known when it was scored, and the index
    filters it out — which is correct, and which would leave this pass silently
    covering nothing if the dates were picked carelessly.
    """
    donation = dataset.donations[-1]
    index = CoverageIndex()
    for source in ("Kompas", "Tempo"):
        index.add(
            CoverageItem(
                entity_id=donation.sender_ref.key,
                source=source,
                published_at=donation.occurred_at - timedelta(days=30),
                headline="Reported",
                url=f"https://example.org/{source.lower()}",
                match_confidence=0.99,
                stage="allegation",
            )
        )
    return index


def _classifier_emissions(vectors):
    """Every wording the classifier lane can produce, over real feature values.

    One lane per feature, ranked on that feature alone, so the pass does not
    stop at whichever three the ranking happens to favour. Values come from the
    generated population rather than being invented, and a feature that is null
    everywhere in it produces nothing — which the caller asserts against, so
    this cannot quietly cover less than it appears to.
    """
    found: list[tuple[str, Lane, str]] = []
    for entry in catalogue():
        if Lane.CLASSIFIER not in entry.lanes or not entry.analyst_facing:
            continue
        name = entry.code.lower()
        lane = ClassifierLane(
            Artifact(
                version="test",
                model=_RankedModel((name,)),
                calibrator=None,
                threshold=0.5,
                feature_names=(name,),
                categorical_features=(),
                manifest={"versions": {"features": "f-test"}},
            )
        )
        for vector in vectors:
            result = lane.evaluate(None, None, vector)
            codes_seen = {reason.code for reason in result.reasons}
            if entry.code not in codes_seen:
                continue
            for reason in result.reasons:
                found.append((reason.code, reason.lane, reason.statement))
            break

    # The fallback, reached by ranking the lane on a quantity it may not state.
    # It is what an analyst sees when the model can name nothing checkable, so
    # leaving it unexercised would leave the likeliest wording unchecked.
    barred = next(e.code.lower() for e in catalogue() if not e.analyst_facing)
    fallback = ClassifierLane(
        Artifact(
            version="test",
            model=_RankedModel((barred,)),
            calibrator=None,
            threshold=0.5,
            feature_names=(barred,),
            categorical_features=(),
            manifest={"versions": {"features": "f-test"}},
        )
    )
    for reason in fallback.evaluate(None, None, vectors[0]).reasons:
        found.append((reason.code, reason.lane, reason.statement))
    return found


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

    def test_the_pass_reaches_every_wording_it_claims_to(self, scored):
        """Guards the check above from passing by covering almost nothing.

        A completeness test that observed four codes out of fifty-eight would
        report the catalogue as verified while leaving most of it unexercised.
        What the pass does not reach is named below rather than left as a
        number nobody looks at, and asserted exactly, so it cannot grow.
        """
        seen = {code for code, _, _ in scored}
        stateable = {entry.code for entry in catalogue() if entry.analyst_facing}
        unexercised = sorted(stateable - seen)
        assert unexercised == sorted(UNREACHED_BY_THIS_POPULATION), (
            f"the wordings this pass leaves unchecked have changed: "
            f"{', '.join(unexercised)}"
        )

    def test_nothing_barred_from_being_stated_was_emitted(self, scored):
        barred = {
            entry.code for entry in catalogue() if not entry.analyst_facing
        }
        assert barred
        assert not barred & {code for code, _, _ in scored}

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

    def test_every_code_shown_to_anybody_carries_wording(self):
        assert all(
            entry.statements and all(s.strip() for s in entry.statements)
            for entry in catalogue()
            if entry.analyst_facing
        )

    def test_a_quantity_with_no_checkable_form_carries_no_wording(self):
        """Catalogued so the enumeration is complete, with nothing to say."""
        barred = [e for e in catalogue() if not e.analyst_facing]
        assert barred
        assert all(entry.statements == () for entry in barred)
        assert all(entry.observation.strip() for entry in barred)

    def test_no_statement_is_a_feature_name_with_its_value_after_it(self):
        """The shape a wording takes when nobody wrote one.

        `amount to limit ratio: 0.42.` names a column of the model's input, not
        anything about the donation, and a reader cannot check a quantity whose
        definition the sentence never states.
        """
        dumps = sorted(
            entry.code
            for entry in catalogue()
            for statement in entry.statements
            if _LABEL_AND_VALUE.fullmatch(statement.strip())
        )
        assert not dumps

    def test_a_label_and_a_value_is_a_defect(self):
        assert wording_defects(_entry("amount to limit ratio: {value}."))
        assert wording_defects(_entry("Amount Log: {value}"))

    def test_a_barred_code_cannot_be_given_wording(self):
        with pytest.raises(ValidationError, match="never shown"):
            ReasonCode(
                code="AMOUNT_LOG",
                lanes=(Lane.CLASSIFIER,),
                source="feature: amount_log",
                observation="a log transform",
                statements=("amount log: {value}.",),
                analyst_facing=False,
            )

    def test_a_code_shown_to_somebody_must_carry_wording(self):
        with pytest.raises(ValidationError, match="no wording"):
            ReasonCode(
                code="EXAMPLE",
                lanes=(Lane.GRAPH,),
                source="RULE-T2-01",
                observation="an example",
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

    def test_the_shipped_ledger_has_a_status_for_every_emitted_code(self, scored):
        """Nothing reaches an analyst whose wording nobody has ruled on.

        This asserted the opposite until the catalogue was reviewed — every
        emitted code unreviewed — and the inversion is the point of keeping it:
        the property worth holding is not which state the ledger is in, but
        that no code can be emitted while sitting outside it. A code added
        tomorrow and rendered into a case bundle without a decision fails here.
        """
        emitted = {code for code, _, _ in scored}
        assert emitted
        statuses = default_statuses()
        assert all(
            statuses.get(code, ReviewStatus.UNREVIEWED) is not ReviewStatus.UNREVIEWED
            for code in emitted
        )


class TestTheClassifierOnlySaysWhatCanBeChecked:
    """The lane emits catalogued wording, and never the barred quantities.

    Structural rather than a matter of review. A feature with no form a reader
    could check is passed over however much weight the model gave it, so it
    cannot reach a case bundle at all — waiting for an analyst to reject the
    wording would leave it in front of people until they did.
    """

    def lane(self, ranking: tuple[str, ...]) -> ClassifierLane:
        return ClassifierLane(
            Artifact(
                version="test",
                model=_RankedModel(ranking),
                calibrator=None,
                threshold=0.5,
                feature_names=ranking,
                categorical_features=(),
                manifest={"versions": {"features": "f-test"}},
            )
        )

    def vector(self, values: dict) -> FeatureVector:
        return FeatureVector(
            donation_id="d-1",
            donation_version=1,
            computed_at=at(2026, 6, 1),
            feature_set_version="f-test",
            values=values,
        )

    def test_a_quantity_with_no_checkable_form_never_becomes_a_reason(self):
        barred = tuple(
            entry.code.lower()
            for entry in catalogue()
            if not entry.analyst_facing
        )
        assert len(barred) >= 3, "the fixture needs barred features to exercise"
        result = self.lane(barred).evaluate(
            None, None, self.vector({name: 1.0 for name in barred})
        )
        assert not undeclared(r.code for r in result.reasons)
        assert all(entry_for(r.code).analyst_facing for r in result.reasons)
        assert {r.code for r in result.reasons} == {"MODEL_SCORE"}

    def test_the_ranking_continues_past_one_it_cannot_state(self):
        """Skipping a quantity costs an explanation, it does not suppress one."""
        result = self.lane(("amount_log", "pair_prior_count")).evaluate(
            None, None, self.vector({"amount_log": 16.1, "pair_prior_count": 3})
        )
        assert [r.code for r in result.reasons] == ["PAIR_PRIOR_COUNT"]
        assert "3 times before" in result.reasons[0].statement

    def test_every_emitted_statement_is_the_catalogued_one(self):
        ranking = ("amount", "amount_to_limit_ratio", "pair_prior_count")
        result = self.lane(ranking).evaluate(
            None,
            None,
            self.vector(
                {
                    "amount": 1_800_000,
                    "amount_to_limit_ratio": 0.42,
                    "pair_prior_count": 3,
                }
            ),
        )
        assert len(result.reasons) == 3
        assert all(
            entry_for(reason.code).matches(reason.statement)
            for reason in result.reasons
        )

    def test_money_is_written_the_way_it_is_read_here(self):
        """A rupiah figure under the other grouping convention is wrong by
        three orders of magnitude."""
        result = self.lane(("amount",)).evaluate(
            None, None, self.vector({"amount": 1_800_000})
        )
        assert result.reasons[0].statement == "This donation is for Rp1.800.000."

    def test_a_share_is_written_as_a_share(self):
        result = self.lane(("amount_to_limit_ratio",)).evaluate(
            None, None, self.vector({"amount_to_limit_ratio": 0.42})
        )
        assert "42%" in result.reasons[0].statement
        assert result.reasons[0].comparison

    def test_a_boolean_that_came_back_false_is_not_a_reason(self):
        """Printing "false" beside a label invites it to be read as one."""
        result = self.lane(("pair_is_first", "pair_prior_count")).evaluate(
            None,
            None,
            self.vector({"pair_is_first": False, "pair_prior_count": 3}),
        )
        assert [r.code for r in result.reasons] == ["PAIR_PRIOR_COUNT"]

    def test_a_boolean_that_came_back_true_states_itself(self):
        result = self.lane(("pair_is_first",)).evaluate(
            None, None, self.vector({"pair_is_first": True})
        )
        assert result.reasons[0].statement == (
            "This is the first donation between these two parties."
        )

    def test_a_feature_the_vector_never_computed_says_nothing(self):
        """A null is a real state the model was trained on. It is not an
        observation, and filling it would report a value nobody measured."""
        result = self.lane(("pair_prior_count",)).evaluate(
            None, None, self.vector({"pair_prior_count": None})
        )
        assert [r.code for r in result.reasons] == ["MODEL_SCORE"]


class _RankedModel:
    """A model that ranks the features it is given, in the order given."""

    def __init__(self, names: tuple[str, ...]) -> None:
        self.feature_importances_ = [
            float(len(names) - index) for index in range(len(names))
        ]

    def predict_proba(self, row):
        import numpy as np

        return np.array([[0.2, 0.8]])
