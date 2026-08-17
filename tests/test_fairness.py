"""Fairness assessment.

What is tested here is mostly that the module refuses to produce numbers it
cannot support. A fairness report that always returns a figure is worse than
none, because the figure is what gets quoted and the caveat is what gets lost.
"""

from __future__ import annotations

import pytest

from cakradana.evaluation.fairness import (
    MAX_FALSE_FLAG_DISPARITY,
    MIN_GROUP_REVIEWED,
    REQUIRED_BREAKDOWNS,
    Cohort,
    affiliation_assessment,
    assess,
    cramers_v,
    differential_performance,
    size_band,
)


def cohorts(
    *,
    group: str,
    attribute: str = "affiliation",
    reviewed_clean: int,
    false_flags: int,
    reviewed_risky: int = 0,
    detected: int = 0,
    start: int = 0,
) -> list[Cohort]:
    """A group with an exactly specified false-flag rate."""
    made: list[Cohort] = []
    index = start
    for position in range(reviewed_clean):
        made.append(
            Cohort(
                donation_id=f"{group}-clean-{index}",
                score=0.5,
                flagged=position < false_flags,
                amount_idr=5_000_000,
                reviewed=True,
                confirmed_risky=False,
                **{attribute: group},
            )
        )
        index += 1
    for position in range(reviewed_risky):
        made.append(
            Cohort(
                donation_id=f"{group}-risky-{index}",
                score=0.9,
                flagged=position < detected,
                amount_idr=5_000_000,
                reviewed=True,
                confirmed_risky=True,
                **{attribute: group},
            )
        )
        index += 1
    return made


class TestSizeBands:
    def test_the_boundaries_follow_the_statutory_limits(self):
        assert size_band(9_999_999) == "under_10jt"
        assert size_band(10_000_000) == "10jt_to_200jt"
        assert size_band(200_000_000) == "200jt_to_2_5m"
        assert size_band(2_500_000_000) == "over_2_5m"

    def test_a_boundary_amount_lands_in_the_upper_band(self):
        """Exactly at a limit is over it for the rule that matters, so the band
        boundary is inclusive on the same side."""
        assert size_band(200_000_000) == "200jt_to_2_5m"


class TestGroupMeasurability:
    def test_a_small_group_gets_no_rate_and_a_reason(self):
        report = differential_performance(
            cohorts(group="A", reviewed_clean=5, false_flags=1), attribute="affiliation"
        )
        group = report.groups[0]
        assert group.false_flag_rate is None
        assert "below the" in (group.unmeasurable_reason or "")

    def test_the_threshold_is_the_documented_one(self):
        just_under = differential_performance(
            cohorts(group="A", reviewed_clean=MIN_GROUP_REVIEWED - 1, false_flags=5),
            attribute="affiliation",
        )
        assert just_under.groups[0].false_flag_rate is None

        at_it = differential_performance(
            cohorts(group="A", reviewed_clean=MIN_GROUP_REVIEWED, false_flags=5),
            attribute="affiliation",
        )
        assert at_it.groups[0].false_flag_rate == pytest.approx(5 / MIN_GROUP_REVIEWED)

    def test_a_group_with_no_confirmed_clean_donations_cannot_be_measured(self):
        """A false-flag rate needs donations that were confirmed fine. Without
        them there is nothing a wrong flag could be wrong about."""
        members = cohorts(
            group="A", reviewed_clean=0, false_flags=0, reviewed_risky=40, detected=40
        )
        report = differential_performance(members, attribute="affiliation")
        group = report.groups[0]
        assert group.false_flag_rate is None
        assert "confirmed clean" in (group.unmeasurable_reason or "")

    def test_an_unreviewed_donation_contributes_no_error_rate(self):
        """Only an adjudicated donation says whether a flag was right."""
        unreviewed = [
            Cohort(
                donation_id=f"u-{i}",
                score=0.9,
                flagged=True,
                affiliation="A",
                reviewed=False,
            )
            for i in range(500)
        ]
        report = differential_performance(unreviewed, attribute="affiliation")
        assert report.groups[0].false_flag_rate is None
        assert report.groups[0].total == 500
        assert report.groups[0].reviewed == 0


class TestDisparity:
    def test_an_even_hand_is_within_tolerance(self):
        members = cohorts(group="A", reviewed_clean=100, false_flags=10) + cohorts(
            group="B", reviewed_clean=100, false_flags=10
        )
        report = differential_performance(members, attribute="affiliation")
        assert report.disparity == pytest.approx(1.0)
        assert report.within_tolerance is True
        assert report.concerns() == ()

    def test_a_group_flagged_in_error_far_more_often_is_reported(self):
        members = cohorts(group="A", reviewed_clean=100, false_flags=30) + cohorts(
            group="B", reviewed_clean=100, false_flags=10
        )
        report = differential_performance(members, attribute="affiliation")
        assert report.disparity == pytest.approx(3.0)
        assert report.within_tolerance is False
        assert report.extremes == ("A", "B")
        assert any("3.00x as often" in concern for concern in report.concerns())

    def test_the_tolerance_is_the_documented_one(self):
        within = cohorts(group="A", reviewed_clean=100, false_flags=10) + cohorts(
            group="B", reviewed_clean=1000, false_flags=80
        )
        report = differential_performance(within, attribute="affiliation")
        assert report.disparity == pytest.approx(MAX_FALSE_FLAG_DISPARITY)
        assert report.within_tolerance is True

    def test_differing_selection_rates_alone_are_not_a_finding(self):
        """One party genuinely receiving more risky donations is a fact about
        the population. Flagging it equally would be the defect."""
        members = cohorts(
            group="A",
            reviewed_clean=100,
            false_flags=10,
            reviewed_risky=100,
            detected=90,
        ) + cohorts(
            group="B", reviewed_clean=100, false_flags=10, reviewed_risky=5, detected=4
        )
        report = differential_performance(members, attribute="affiliation")
        rates = {g.group: g.selection_rate for g in report.groups}
        assert rates["A"] > rates["B"] * 2
        assert report.within_tolerance is True
        assert report.concerns() == ()

    def test_one_measurable_group_yields_no_disparity(self):
        members = cohorts(group="A", reviewed_clean=100, false_flags=10) + cohorts(
            group="B", reviewed_clean=3, false_flags=1
        )
        report = differential_performance(members, attribute="affiliation")
        assert report.disparity is None
        assert report.within_tolerance is None
        assert "at least two" in (report.unmeasurable_reason or "")

    def test_a_zero_baseline_is_unmeasurable_rather_than_infinite(self):
        """"Infinitely worse" is not a finding anyone can act on."""
        members = cohorts(group="A", reviewed_clean=100, false_flags=10) + cohorts(
            group="B", reviewed_clean=100, false_flags=0
        )
        report = differential_performance(members, attribute="affiliation")
        assert report.disparity is None
        assert "no denominator" in (report.unmeasurable_reason or "")

    def test_unknown_attributes_form_their_own_group_rather_than_vanishing(self):
        """If the unknowns are one district whose records digitise badly,
        dropping them hides exactly that."""
        members = cohorts(group="A", reviewed_clean=100, false_flags=10)
        members += [
            Cohort(
                donation_id=f"x-{i}",
                score=0.5,
                flagged=i < 40,
                district=None,
                reviewed=True,
                confirmed_risky=False,
            )
            for i in range(100)
        ]
        report = differential_performance(members, attribute="district")
        names = {g.group for g in report.groups}
        assert "unknown" in names
        assert sum(g.total for g in report.groups) == len(members)

    def test_a_size_band_breakdown_uses_the_amount(self):
        members = [
            Cohort(
                donation_id=f"s-{i}",
                score=0.5,
                flagged=False,
                amount_idr=5_000_000_000,
                reviewed=True,
            )
            for i in range(40)
        ]
        report = differential_performance(members, attribute="size_band")
        assert report.groups[0].group == "over_2_5m"


class TestAssociation:
    def test_no_association_when_flagging_is_even(self):
        members = cohorts(group="A", reviewed_clean=100, false_flags=50) + cohorts(
            group="B", reviewed_clean=100, false_flags=50
        )
        assert cramers_v(members) == pytest.approx(0.0, abs=1e-9)

    def test_perfect_separation_scores_one(self):
        members = cohorts(group="A", reviewed_clean=100, false_flags=100) + cohorts(
            group="B", reviewed_clean=100, false_flags=0
        )
        assert cramers_v(members) == pytest.approx(1.0)

    def test_a_single_party_has_nothing_to_associate_with(self):
        assert cramers_v(cohorts(group="A", reviewed_clean=100, false_flags=50)) is None

    def test_nothing_flagged_is_not_an_association_of_zero(self):
        """A degenerate table has no association, measured or otherwise."""
        members = cohorts(group="A", reviewed_clean=100, false_flags=0) + cohorts(
            group="B", reviewed_clean=100, false_flags=0
        )
        assert cramers_v(members) is None

    def test_the_scale_does_not_grow_with_sample_size(self):
        """Chi-square would. That is why it is not the reported figure."""
        small = cohorts(group="A", reviewed_clean=50, false_flags=30) + cohorts(
            group="B", reviewed_clean=50, false_flags=20
        )
        large = cohorts(group="A", reviewed_clean=5000, false_flags=3000) + cohorts(
            group="B", reviewed_clean=5000, false_flags=2000
        )
        assert cramers_v(small) == pytest.approx(cramers_v(large), abs=1e-6)


class TestAffiliationAssessment:
    def test_an_even_hand_across_parties_is_acceptable(self):
        members = cohorts(group="PartaiA", reviewed_clean=200, false_flags=20) + cohorts(
            group="PartaiB", reviewed_clean=200, false_flags=20
        )
        report = affiliation_assessment(members)
        assert report.acceptable is True
        assert set(report.selection_rates) == {"PartaiA", "PartaiB"}

    def test_a_disparate_error_rate_is_not_acceptable(self):
        members = cohorts(group="PartaiA", reviewed_clean=200, false_flags=60) + cohorts(
            group="PartaiB", reviewed_clean=200, false_flags=20
        )
        assert affiliation_assessment(members).acceptable is False

    def test_widespread_unknown_affiliation_is_reported(self):
        members = cohorts(group="PartaiA", reviewed_clean=100, false_flags=10) + cohorts(
            group="PartaiB", reviewed_clean=100, false_flags=10
        )
        members += [
            Cohort(donation_id=f"n-{i}", score=0.4, flagged=False, reviewed=True)
            for i in range(200)
        ]
        report = affiliation_assessment(members)
        assert report.unknown_affiliation_share == pytest.approx(0.5)
        assert any("not the population" in c for c in report.concerns())

    def test_an_empty_population_measures_nothing(self):
        report = affiliation_assessment([])
        assert report.acceptable is None
        assert report.unmeasurable_reason


class TestFullAssessment:
    #: One amount per band, so every required breakdown has enough members to
    #: measure. A population that populates only one band leaves that dimension
    #: unassessed, which is a different verdict from passing.
    AMOUNTS = (5_000_000, 50_000_000, 500_000_000, 5_000_000_000)

    def population(self, *, district_bias: bool = False) -> list[Cohort]:
        made: list[Cohort] = []
        districts = ("Jakarta", "Surabaya")
        for index in range(800):
            district = districts[index % 2]
            biased = district_bias and district == "Jakarta"
            made.append(
                Cohort(
                    donation_id=f"d-{index}",
                    score=0.5,
                    # 7 is coprime with every modulus below, so the flags fall
                    # evenly across all four breakdowns rather than piling into
                    # one group and manufacturing a disparity.
                    flagged=(index % 7 == 0) or (biased and index % 3 == 0),
                    affiliation=f"Partai{'AB'[index % 2]}",
                    district=district,
                    recipient_type="candidate" if index % 3 else "party",
                    amount_idr=self.AMOUNTS[index % 4],
                    reviewed=True,
                    confirmed_risky=False,
                )
            )
        return made

    def test_every_required_dimension_is_assessed(self):
        report = assess(self.population())
        assert tuple(r.attribute for r in report.breakdowns) == REQUIRED_BREAKDOWNS

    def test_a_clean_population_passes(self):
        assert assess(self.population()).passed is True

    def test_a_district_disparity_fails_the_whole_assessment(self):
        report = assess(self.population(district_bias=True))
        assert report.passed is False
        assert any("district" in concern for concern in report.concerns())

    def test_an_unmeasurable_dimension_is_none_not_false(self):
        """"Nobody knows" and "a disparity was found" both block, but they call
        for different responses, so they are not the same value."""
        report = assess(
            cohorts(group="PartaiA", reviewed_clean=100, false_flags=10)
            + cohorts(group="PartaiB", reviewed_clean=100, false_flags=10)
        )
        assert report.passed is None

    def test_a_disparity_outranks_an_unmeasured_dimension(self):
        report = assess(
            cohorts(group="PartaiA", reviewed_clean=200, false_flags=80)
            + cohorts(group="PartaiB", reviewed_clean=200, false_flags=20)
        )
        assert report.passed is False

    def test_the_description_names_the_verdict(self):
        assert "fairness:" in assess(self.population()).describe()
