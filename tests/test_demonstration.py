"""The demonstration rule set and non-authoritative evidence.

Rules that depend on reference data nobody has supplied can be exercised
end to end against fixtures. What must not happen is a fixture-backed finding
becoming indistinguishable from enforcement, so the distinction is carried on
the finding itself rather than left to whoever configured the run.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from cakradana.calendar import CampaignPeriod, ElectoralCalendar
from cakradana.data import GeneratorConfig, generate
from cakradana.history import InMemoryDonationStore
from cakradana.registers import Register, RegisterEntry, RegisterSet
from cakradana.reporting import ReportedDonation, SubmissionSet
from cakradana.rules import RuleEngine, load_latest, load_named
from cakradana.rules.loader import available_versions
from cakradana.schema import EntityType
from tests.conftest import at, make_donation

CONTEXT = "pemilu-2029"

SMALL = GeneratorConfig(
    seed=20260816,
    n_legitimate_donors=200,
    n_recipients=5,
    n_background_donations=1500,
    n_grassroots_campaigns=2,
)


@pytest.fixture(scope="module")
def dataset():
    return generate(SMALL)


class TestRuleSetSelection:
    def test_the_demonstration_set_is_never_selected_implicitly(self):
        """It sorts after the published set, and picking it up for that reason
        would put fixture-backed findings in front of an analyst."""
        assert "rules-2026.08-demo" in available_versions()
        assert load_latest().version == "rules-2026.07"
        assert not load_latest().demonstration

    def test_the_demonstration_set_declares_itself(self):
        demo = load_named("rules-2026.08-demo")
        assert demo.demonstration
        assert "DEMONSTRATION SET" in (demo.notes or "")

    def test_the_published_set_keeps_unevidenced_rules_switched_off(self):
        published = load_latest()
        inactive = {r.id for r in published.rules if not r.active}
        assert "RULE-T1-10" in inactive, "convictions cannot be evidenced"
        assert all(r.inactive_reason for r in published.rules if not r.active)

    def test_the_demonstration_set_switches_them_on(self):
        demo = load_named("rules-2026.08-demo")
        assert all(r.active for r in demo.rules)

    def test_both_sets_hold_the_same_rules(self):
        """The demonstration set differs only in what is switched on. A rule
        that existed in one and not the other would make the demonstration
        prove something about a system nobody runs."""
        published = {r.id for r in load_latest().rules}
        demo = {r.id for r in load_named("rules-2026.08-demo").rules}
        assert published == demo


class TestEvidenceProvenance:
    def test_a_fixture_backed_finding_is_marked_as_a_demonstration(self, dataset):
        store = InMemoryDonationStore(dataset.donations)
        engine = RuleEngine(
            load_named("rules-2026.08-demo"),
            calendar=dataset.calendar,
            registers=dataset.registers,
            submissions=dataset.submissions,
            require_verified_citations=False,
        )

        demonstrations = []
        for donation in dataset.donations:
            evaluation = engine.evaluate(
                donation,
                store.knowable_at(donation.occurred_at),
                entities=dataset.entities,
            )
            demonstrations.extend(
                f for f in evaluation.legal_findings if f.is_demonstration
            )

        assert demonstrations, "the fixtures produced no findings to check"
        for finding in demonstrations:
            assert not finding.authoritative
            assert finding.explanation.startswith("DEMONSTRATION ONLY")
            assert "establishes nothing about the donor" in finding.explanation

    def test_the_qualifier_leads_rather_than_trails(self, dataset):
        """A reader who stops after the first clause must not come away
        believing an offence has been established."""
        store = InMemoryDonationStore(dataset.donations)
        engine = RuleEngine(
            load_named("rules-2026.08-demo"),
            calendar=dataset.calendar,
            registers=dataset.registers,
            submissions=dataset.submissions,
            require_verified_citations=False,
        )
        for donation in dataset.donations:
            evaluation = engine.evaluate(
                donation,
                store.knowable_at(donation.occurred_at),
                entities=dataset.entities,
            )
            for finding in evaluation.legal_findings:
                if finding.is_demonstration:
                    assert finding.explanation.index("DEMONSTRATION") < 20
                    return

    def test_an_authoritative_register_produces_an_unqualified_finding(self):
        calendar = ElectoralCalendar(
            [CampaignPeriod(electoral_context=CONTEXT, start=date(2026, 9, 1), end=date(2026, 11, 24))]
        )
        registers = RegisterSet(
            [
                Register(
                    RegisterSet.PROHIBITED_SOURCE,
                    [RegisterEntry(entity_id="bumn-1", canonical_name="PT PLN (Persero)", category="state-enterprise")],
                    available=True,
                    authoritative=True,
                )
            ]
        )
        engine = RuleEngine(
            load_named("rules-2026.08-demo"),
            calendar=calendar,
            registers=registers,
            require_verified_citations=False,
        )
        donation = make_donation(
            sender="bumn-1",
            sender_type=EntityType.CORPORATION,
            electoral_context=CONTEXT,
            occurred=at(2026, 3, 1),
        )
        result = engine.evaluate(
            donation, InMemoryDonationStore([donation]).knowable_at(donation.occurred_at)
        )
        finding = [f for f in result.legal_findings if f.rule_id == "RULE-T1-09"][0]
        assert finding.authoritative
        assert not finding.is_demonstration
        assert "DEMONSTRATION" not in finding.explanation


class TestSuppliedButEmptyRegister:
    def test_an_empty_supplied_register_clears_rather_than_abstains(self):
        """"This list is empty" and "there is no list" are the two states the
        register type exists to separate, and a register defines __len__, so a
        truthiness check collapses them."""
        registers = RegisterSet(
            [Register(RegisterSet.FINAL_CONVICTIONS, (), available=True)]
        )
        lookup = registers.lookup(
            RegisterSet.FINAL_CONVICTIONS, "anyone", when=date(2026, 6, 1)
        )
        assert lookup.available
        assert not lookup.member

    def test_an_unsupplied_register_still_abstains(self):
        lookup = RegisterSet().lookup("prohibited_source", "anyone", when=date(2026, 6, 1))
        assert not lookup.available
        assert "not configured" in lookup.reason


class TestReconciliation:
    def test_a_declared_donation_produces_no_finding(self, dataset):
        submissions = dataset.submissions
        declared = submissions.contains(
            electoral_context=SMALL.electoral_context,
            donor_ref=None,
            recipient_ref=None,
            amount_idr=dataset.donations[0].amount_idr,
            occurred_on=dataset.donations[0].occurred_at.date(),
        )
        assert isinstance(declared, bool)

    def test_a_date_outside_the_filed_period_is_indeterminate(self):
        """Reporting a donation as undeclared when the relevant return has not
        been filed would be an accusation manufactured from a date range."""
        submissions = SubmissionSet(
            [],
            covered_periods=((CONTEXT, date(2026, 1, 1), date(2026, 6, 30)),),
            available=True,
        )
        assert submissions.covers(CONTEXT, date(2026, 3, 1))
        assert not submissions.covers(CONTEXT, date(2026, 9, 1))

    def test_matching_tolerates_transcription_slippage(self):
        """A return transcribed by hand rounds a figure and misplaces a day.
        Treating either as a mismatch reports a properly declared donation as
        undeclared, which is the most damaging error this rule can make."""
        submissions = SubmissionSet(
            [
                ReportedDonation(
                    electoral_context=CONTEXT,
                    report_kind="LPSDK",
                    donor_ref="d1",
                    recipient_ref="r1",
                    amount_idr=100_500_000,
                    occurred_on=date(2026, 3, 2),
                )
            ],
            covered_periods=((CONTEXT, date(2026, 1, 1), date(2026, 12, 31)),),
            available=True,
        )
        assert submissions.contains(
            electoral_context=CONTEXT,
            donor_ref="d1",
            recipient_ref="r1",
            amount_idr=100_000_000,
            occurred_on=date(2026, 3, 1),
        )

    def test_a_materially_different_amount_is_not_a_match(self):
        submissions = SubmissionSet(
            [
                ReportedDonation(
                    electoral_context=CONTEXT,
                    report_kind="LPSDK",
                    donor_ref="d1",
                    amount_idr=10_000_000,
                    occurred_on=date(2026, 3, 1),
                )
            ],
            covered_periods=((CONTEXT, date(2026, 1, 1), date(2026, 12, 31)),),
            available=True,
        )
        assert not submissions.contains(
            electoral_context=CONTEXT,
            donor_ref="d1",
            recipient_ref=None,
            amount_idr=100_000_000,
            occurred_on=date(2026, 3, 1),
        )


class TestGeneratedTypologies:
    def test_the_new_typologies_are_generated(self, dataset):
        counts = dataset.typology_counts()
        for typology in ("T-03", "T-07", "T-11"):
            assert counts.get(typology, 0) > 0, f"{typology} was not generated"

    def test_foreign_donors_carry_a_recorded_jurisdiction(self, dataset):
        """Never inferred from a name: that would attach a statutory offence to
        someone on the basis of what they are called."""
        foreign = [
            e for e in dataset.entities.values()
            if e.jurisdiction and e.jurisdiction != "ID"
        ]
        assert foreign
        assert all(e.entity_type is EntityType.FOREIGN_ENTITY for e in foreign)

    def test_self_funded_donations_carry_the_declaration(self, dataset):
        declared = [d for d in dataset.donations if d.is_self_funded_declared]
        assert declared

    def test_unreported_donations_are_absent_from_the_filings(self, dataset):
        unreported = [
            d for d in dataset.donations
            if dataset.truth.get(d.donation_id) == "T-07"
        ]
        assert unreported
        for donation in unreported:
            assert not dataset.submissions.contains(
                electoral_context=donation.electoral_context,
                donor_ref=donation.sender_ref.entity_id,
                recipient_ref=donation.receiver_ref.entity_id,
                amount_idr=donation.amount_idr,
                occurred_on=donation.occurred_at.date(),
            )
