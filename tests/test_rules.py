"""Rule engine behaviour.

The tests that matter most here assert what the engine refuses to do. A rule
that produces a confident wrong answer on missing data is worse than one that
produces none, because the wrong answer is indistinguishable from a real
finding once it reaches an analyst.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from cakradana.calendar import CampaignPeriod, ElectoralCalendar
from cakradana.history import InMemoryDonationStore
from cakradana.registers import Register, RegisterEntry, RegisterSet
from cakradana.rules import LimitTable, RuleEngine, load_latest
from cakradana.rules.schema import (
    EffectiveWindow,
    LegalBasis,
    Rule,
    RuleOutcomeSpec,
    RuleSet,
    RuleTest,
)
from cakradana.schema import Channel, Donation, EntityRef, EntityType
from cakradana.schema.enums import Regime, RuleOutcome
from tests.conftest import at, make_donation

INDIVIDUAL_PARTY_LIMIT = 200_000_000
COMPANY_PARTY_LIMIT = 800_000_000
INDIVIDUAL_CAMPAIGN_LIMIT = 2_500_000_000
COMPANY_CAMPAIGN_LIMIT = 25_000_000_000

CONTEXT = "pemilu-2029"


@pytest.fixture(scope="session")
def ruleset() -> RuleSet:
    return load_latest()


@pytest.fixture
def calendar() -> ElectoralCalendar:
    return ElectoralCalendar(
        [
            CampaignPeriod(
                electoral_context=CONTEXT,
                start=date(2028, 11, 1),
                end=date(2029, 2, 14),
                reporting_deadlines=(date(2029, 1, 15),),
            )
        ]
    )


@pytest.fixture
def engine(ruleset, calendar) -> RuleEngine:
    # Citation verification is disabled here because these tests exercise rule
    # mechanics against synthetic data. Against real data it stays on, and an
    # unverified statutory rule reports indeterminate.
    return RuleEngine(ruleset, calendar=calendar, require_verified_citations=False)


def party_donation(**kwargs) -> Donation:
    kwargs.setdefault("electoral_context", CONTEXT)
    kwargs.setdefault("occurred", at(2026, 3, 1))
    return make_donation(**kwargs)


def evaluate(engine: RuleEngine, donation: Donation, store=None):
    store = store or InMemoryDonationStore([donation])
    return engine.evaluate(donation, store.knowable_at(donation.occurred_at))


class TestRuleSetIntegrity:
    def test_ruleset_loads(self, ruleset):
        assert len(ruleset.tier1) == 11
        assert len(ruleset.tier2) == 10

    def test_statutory_rules_cannot_be_training_labels(self):
        with pytest.raises(ValueError, match="not training labels"):
            Rule(
                id="X",
                tier=1,
                title="t",
                legal_basis=LegalBasis(statute="s", article="a"),
                effective=EffectiveWindow(**{"from": date(2011, 1, 1)}),
                test=RuleTest(kind="threshold"),
                outcome=RuleOutcomeSpec(
                    result=RuleOutcome.LEGAL_FINDING, label_weight=0.5
                ),
            )

    def test_behavioural_rules_cannot_cite_a_statute(self):
        """A heuristic presented with a citation would misstate its standing."""
        with pytest.raises(ValueError, match="no legal basis"):
            Rule(
                id="X",
                tier=2,
                title="t",
                legal_basis=LegalBasis(statute="s", article="a"),
                effective=EffectiveWindow(**{"from": date(2011, 1, 1)}),
                test=RuleTest(kind="fan_out"),
                outcome=RuleOutcomeSpec(result=RuleOutcome.BEHAVIOURAL_SIGNAL),
            )

    def test_behavioural_rules_cannot_produce_legal_findings(self):
        with pytest.raises(ValueError, match="must not produce a"):
            Rule(
                id="X",
                tier=2,
                title="t",
                effective=EffectiveWindow(**{"from": date(2011, 1, 1)}),
                test=RuleTest(kind="fan_out"),
                outcome=RuleOutcomeSpec(result=RuleOutcome.LEGAL_FINDING),
            )

    def test_unknown_test_kinds_are_rejected_at_load(self, ruleset):
        from cakradana.rules.loader import _reject_unknown_tests

        broken = ruleset.model_copy(
            update={
                "rules": (
                    ruleset.rules[0].model_copy(
                        update={"test": RuleTest(kind="not_a_real_test")}
                    ),
                )
            }
        )
        with pytest.raises(ValueError, match="unknown tests"):
            _reject_unknown_tests(broken)


class TestLimitTable:
    def test_limits_derive_from_the_rules_that_cite_them(self, ruleset):
        limits = LimitTable.from_ruleset(ruleset)
        assert (
            limits.for_donation(Regime.PARTY_ANNUAL, EntityType.INDIVIDUAL).amount_idr
            == INDIVIDUAL_PARTY_LIMIT
        )
        assert (
            limits.for_donation(Regime.PARTY_ANNUAL, EntityType.CORPORATION).amount_idr
            == COMPANY_PARTY_LIMIT
        )
        assert (
            limits.for_donation(Regime.CAMPAIGN, EntityType.INDIVIDUAL).amount_idr
            == INDIVIDUAL_CAMPAIGN_LIMIT
        )
        assert (
            limits.for_donation(Regime.CAMPAIGN, EntityType.CORPORATION).amount_idr
            == COMPANY_CAMPAIGN_LIMIT
        )

    def test_unknown_donor_type_gets_no_limit(self, ruleset):
        """Falling through to the organisation limit would judge an
        unidentified donor against the most permissive cap available."""
        limits = LimitTable.from_ruleset(ruleset)
        assert limits.for_donation(Regime.PARTY_ANNUAL, EntityType.UNKNOWN) is None

    def test_indeterminate_regime_gets_no_limit(self, ruleset):
        limits = LimitTable.from_ruleset(ruleset)
        assert limits.for_donation(Regime.INDETERMINATE, EntityType.INDIVIDUAL) is None


class TestSingleTransactionLimits:
    def test_donation_at_the_cap_passes(self, engine):
        result = evaluate(engine, party_donation(amount_idr=INDIVIDUAL_PARTY_LIMIT))
        assert not result.legal_findings

    def test_donation_over_the_cap_is_a_finding(self, engine):
        result = evaluate(engine, party_donation(amount_idr=INDIVIDUAL_PARTY_LIMIT + 1))
        findings = {f.rule_id: f for f in result.legal_findings}
        assert "RULE-T1-01" in findings
        assert findings["RULE-T1-01"].threshold_idr == INDIVIDUAL_PARTY_LIMIT
        assert "Pasal" in findings["RULE-T1-01"].article
        # A single donation above the cap is also, trivially, a cumulative
        # breach of it, so both rules reporting is correct rather than double
        # counting.
        assert "RULE-T1-05" in findings

    def test_company_is_judged_against_the_company_cap(self, engine):
        under = party_donation(
            amount_idr=COMPANY_PARTY_LIMIT - 1, sender_type=EntityType.CORPORATION
        )
        over = party_donation(
            amount_idr=COMPANY_PARTY_LIMIT + 1, sender_type=EntityType.CORPORATION
        )
        assert not evaluate(engine, under).legal_findings
        assert "RULE-T1-02" in {
            f.rule_id for f in evaluate(engine, over).legal_findings
        }


class TestCumulativeLimits:
    """The gap that motivated this rule.

    The statutory caps are per donor per period. The previous implementation
    compared a single row's amount against the cap, so a donor splitting a
    contribution passed every check while breaching the cap many times over.
    """

    def test_repeated_donations_at_the_cap_are_caught(self, engine):
        store = InMemoryDonationStore()
        donations = []
        for i in range(20):
            d = party_donation(
                donation_id=f"d{i}",
                amount_idr=INDIVIDUAL_PARTY_LIMIT,
                occurred=at(2026, 3, 1) + timedelta(days=7 * i),
            )
            donations.append(d)
            store.add(d)

        findings_at = [
            i
            for i, d in enumerate(donations)
            if evaluate(engine, d, store).legal_findings
        ]
        assert findings_at[0] == 1, "the second donation is the one that crosses"

        # And the single-transaction rule never fires on any of them.
        for d in donations:
            single = [
                r for r in evaluate(engine, d, store).results if r.rule_id == "RULE-T1-01"
            ][0]
            assert single.outcome is RuleOutcome.PASS

    def test_finding_names_the_running_total_and_the_period(self, engine):
        store = InMemoryDonationStore()
        for i in range(3):
            store.add(
                party_donation(
                    donation_id=f"d{i}",
                    amount_idr=100_000_000,
                    occurred=at(2026, 3, 1) + timedelta(days=i),
                )
            )
        last = party_donation(
            donation_id="d-last",
            amount_idr=100_000_000,
            occurred=at(2026, 3, 10),
        )
        store.add(last)
        finding = [
            f for f in evaluate(engine, last, store).legal_findings
            if f.rule_id == "RULE-T1-05"
        ][0]
        assert finding.observed == 400_000_000
        assert "2026" in finding.explanation
        assert "Rp400.000.000" in finding.explanation

    def test_donations_in_a_later_period_do_not_accumulate(self, engine):
        store = InMemoryDonationStore()
        store.add(
            party_donation(
                donation_id="d-2025",
                amount_idr=INDIVIDUAL_PARTY_LIMIT,
                occurred=at(2025, 12, 1),
            )
        )
        current = party_donation(
            donation_id="d-2026",
            amount_idr=INDIVIDUAL_PARTY_LIMIT,
            occurred=at(2026, 1, 5),
        )
        store.add(current)
        assert not evaluate(engine, current, store).legal_findings

    def test_unresolved_donor_cannot_be_accumulated(self, engine):
        """The rule refuses rather than grouping by name, because splitting a
        donor across name variants is the behaviour it exists to catch."""
        donation = Donation(
            donation_id="d-1",
            sender_ref=EntityRef(raw_text="Budi Santoso", entity_type=EntityType.INDIVIDUAL),
            receiver_ref=EntityRef(entity_id="party-1", entity_type=EntityType.POLITICAL_PARTY),
            amount_idr=50_000_000,
            occurred_at=at(2026, 3, 1),
            recorded_at=at(2026, 3, 1),
            channel=Channel.WEB_SCRAPE,
            electoral_context=CONTEXT,
        )
        result = evaluate(engine, donation)
        cumulative = [r for r in result.results if r.rule_id == "RULE-T1-05"][0]
        assert cumulative.outcome is RuleOutcome.INDETERMINATE
        assert "resolved_sender_id" in cumulative.reason


class TestPointInTime:
    def test_a_later_recorded_donation_does_not_change_an_earlier_score(self, engine):
        """A donation scraped in June was not knowable in February. Letting it
        into a February aggregate is the leak that invalidated the earlier
        measurements of this system."""
        early = party_donation(
            donation_id="early", amount_idr=150_000_000, occurred=at(2026, 2, 1)
        )
        late_arrival = party_donation(
            donation_id="late",
            amount_idr=150_000_000,
            occurred=at(2026, 1, 15),
            recorded=at(2026, 6, 1),
        )
        store = InMemoryDonationStore([early, late_arrival])
        assert not evaluate(engine, early, store).legal_findings

    def test_the_same_donation_is_visible_once_recorded(self, engine):
        late_arrival = party_donation(
            donation_id="late",
            amount_idr=150_000_000,
            occurred=at(2026, 1, 15),
            recorded=at(2026, 6, 1),
        )
        later = party_donation(
            donation_id="later", amount_idr=150_000_000, occurred=at(2026, 7, 1)
        )
        store = InMemoryDonationStore([late_arrival, later])
        assert evaluate(engine, later, store).legal_findings


class TestIndeterminacy:
    def test_unknown_electoral_context_yields_no_limit_findings(self, engine):
        """Assuming the annual regime would judge a lawful campaign donation
        against a cap an order of magnitude lower and manufacture a false
        statutory finding."""
        donation = make_donation(
            amount_idr=1_000_000_000, electoral_context="unconfigured-election"
        )
        result = evaluate(engine, donation)
        assert not result.legal_findings
        assert any(r.rule_id == "RULE-T1-01" for r in result.indeterminate)
        assert not result.is_fully_evaluated

    def test_unavailable_register_is_indeterminate_not_a_pass(self, engine):
        result = evaluate(engine, party_donation())
        prohibited = [r for r in result.results if r.rule_id == "RULE-T1-09"][0]
        assert prohibited.outcome is RuleOutcome.INDETERMINATE
        assert "register" in prohibited.reason

    def test_inactive_rules_report_why(self, engine):
        result = evaluate(engine, party_donation())
        for rule_id in ("RULE-T1-10", "RULE-T1-11", "RULE-T2-05"):
            found = [r for r in result.results if r.rule_id == rule_id][0]
            assert found.outcome is RuleOutcome.INDETERMINATE
            assert found.reason

    def test_unverified_citations_block_statutory_findings(self, ruleset, calendar):
        """Consistency across source documents is not verification, and a
        statutory finding names an article to a subject."""
        strict = RuleEngine(ruleset, calendar=calendar, require_verified_citations=True)
        donation = party_donation(amount_idr=INDIVIDUAL_PARTY_LIMIT * 5)
        result = strict.evaluate(
            donation, InMemoryDonationStore([donation]).knowable_at(donation.occurred_at)
        )
        assert not result.legal_findings
        assert any("not been verified" in (r.reason or "") for r in result.indeterminate)

    def test_foreign_source_is_indeterminate_without_jurisdiction(self, engine):
        result = evaluate(engine, party_donation())
        foreign = [r for r in result.results if r.rule_id == "RULE-T1-07"][0]
        assert foreign.outcome is RuleOutcome.INDETERMINATE
        assert "inferred from a name" in foreign.reason


class TestProhibitedSourceRegister:
    def test_a_supplied_register_produces_a_finding(self, ruleset, calendar):
        registers = RegisterSet(
            [
                Register(
                    RegisterSet.PROHIBITED_SOURCE,
                    [
                        RegisterEntry(
                            entity_id="bumn-1",
                            canonical_name="PT PLN (Persero)",
                            category="state-enterprise",
                        )
                    ],
                    available=True,
                )
            ]
        )
        engine = RuleEngine(
            ruleset,
            calendar=calendar,
            registers=registers,
            require_verified_citations=False,
        )
        donation = party_donation(sender="bumn-1", sender_type=EntityType.CORPORATION)
        result = engine.evaluate(
            donation,
            InMemoryDonationStore([donation]).knowable_at(donation.occurred_at),
        )
        finding = [f for f in result.legal_findings if f.rule_id == "RULE-T1-09"]
        assert finding and "state-enterprise" in finding[0].explanation

    def test_a_similarly_named_company_is_not_matched(self, ruleset, calendar):
        """Name prefixes do not separate a state enterprise from an ordinary
        company, so only an actual register entry produces a finding."""
        registers = RegisterSet(
            [
                Register(
                    RegisterSet.PROHIBITED_SOURCE,
                    [RegisterEntry(entity_id="bumn-1", canonical_name="PT PLN (Persero)")],
                    available=True,
                )
            ]
        )
        engine = RuleEngine(
            ruleset,
            calendar=calendar,
            registers=registers,
            require_verified_citations=False,
        )
        donation = party_donation(
            sender="company-9", sender_type=EntityType.CORPORATION
        )
        result = engine.evaluate(
            donation,
            InMemoryDonationStore([donation]).knowable_at(donation.occurred_at),
        )
        assert not [f for f in result.legal_findings if f.rule_id == "RULE-T1-09"]


class TestBehaviouralSignals:
    def test_fan_in_burst_fires_on_many_new_donors(self, engine):
        store = InMemoryDonationStore()
        donations = []
        for i in range(18):
            d = party_donation(
                donation_id=f"d{i}",
                sender=f"donor-{i}",
                amount_idr=10_000_000,
                occurred=at(2026, 3, 1) + timedelta(hours=6 * i),
            )
            donations.append(d)
            store.add(d)
        result = evaluate(engine, donations[-1], store)
        signals = {s.rule_id for s in result.behavioural_signals}
        assert "RULE-T2-01" in signals

    def test_fan_in_does_not_fire_on_established_donors(self, engine):
        """Genuine grassroots fundraising also produces fan-in. Without the
        thin-donor condition this test would flag every successful appeal."""
        store = InMemoryDonationStore()
        for i in range(18):
            store.add(
                party_donation(
                    donation_id=f"hist-{i}",
                    sender=f"donor-{i}",
                    amount_idr=5_000_000,
                    occurred=at(2025, 6, 1) + timedelta(days=i),
                )
            )
        donations = []
        for i in range(18):
            d = party_donation(
                donation_id=f"d{i}",
                sender=f"donor-{i}",
                amount_idr=10_000_000,
                occurred=at(2026, 3, 1) + timedelta(hours=6 * i),
            )
            donations.append(d)
            store.add(d)
        result = evaluate(engine, donations[-1], store)
        assert "RULE-T2-01" not in {s.rule_id for s in result.behavioural_signals}

    def test_structuring_band_fires_just_below_the_limit(self, engine):
        just_under = party_donation(amount_idr=int(INDIVIDUAL_PARTY_LIMIT * 0.97))
        result = evaluate(engine, just_under)
        assert "RULE-T2-02" in {s.rule_id for s in result.behavioural_signals}

    def test_structuring_band_ignores_ordinary_amounts(self, engine):
        ordinary = party_donation(amount_idr=int(INDIVIDUAL_PARTY_LIMIT * 0.2))
        result = evaluate(engine, ordinary)
        assert "RULE-T2-02" not in {s.rule_id for s in result.behavioural_signals}

    def test_behavioural_signals_carry_a_label_weight_below_human_judgement(
        self, engine
    ):
        just_under = party_donation(amount_idr=int(INDIVIDUAL_PARTY_LIMIT * 0.97))
        result = evaluate(engine, just_under)
        assert 0 < result.tier2_label_weight() < 0.9

    def test_no_behavioural_signal_carries_a_citation(self, engine):
        store = InMemoryDonationStore()
        donations = []
        for i in range(18):
            d = party_donation(
                donation_id=f"d{i}",
                sender=f"donor-{i}",
                amount_idr=10_000_000,
                occurred=at(2026, 3, 1) + timedelta(hours=6 * i),
            )
            donations.append(d)
            store.add(d)
        for signal in evaluate(engine, donations[-1], store).behavioural_signals:
            assert signal.statute is None
            assert signal.article is None


class TestEffectiveDating:
    def test_a_donation_is_judged_by_the_rules_in_force_on_its_own_date(
        self, ruleset, calendar
    ):
        engine = RuleEngine(
            ruleset, calendar=calendar, require_verified_citations=False
        )
        # The campaign-limit rules take effect in 2017; a 2015 donation is not
        # evaluated against them.
        old = make_donation(
            donation_id="old",
            amount_idr=INDIVIDUAL_PARTY_LIMIT + 1,
            occurred=at(2015, 5, 1),
            electoral_context=CONTEXT,
        )
        result = engine.evaluate(
            old, InMemoryDonationStore([old]).knowable_at(old.occurred_at)
        )
        assert not [r for r in result.results if r.rule_id == "RULE-T1-03"]
        assert [r for r in result.results if r.rule_id == "RULE-T1-01"]
