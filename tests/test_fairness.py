"""Fairness assessment.

What is tested here is mostly that the module refuses to produce numbers it
cannot support. A fairness report that always returns a figure is worse than
none, because the figure is what gets quoted and the caveat is what gets lost.
"""

from __future__ import annotations

import pytest

from cakradana.evaluation.fairness import (
    MAX_FALSE_FLAG_DISPARITY,
    MIN_GROUP_CLEAN,
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
            cohorts(group="A", reviewed_clean=MIN_GROUP_CLEAN - 1, false_flags=5),
            attribute="affiliation",
        )
        assert just_under.groups[0].false_flag_rate is None

        at_it = differential_performance(
            cohorts(group="A", reviewed_clean=MIN_GROUP_CLEAN, false_flags=5),
            attribute="affiliation",
        )
        assert at_it.groups[0].false_flag_rate == pytest.approx(5 / MIN_GROUP_CLEAN)

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
        # The known half has to carry the attribute being broken down on, or
        # every member lands in "unknown" and the assertion is trivially true.
        members = cohorts(
            group="Jakarta", attribute="district", reviewed_clean=100, false_flags=10
        )
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
        assert names == {"Jakarta", "unknown"}
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


class TestDefectsFoundInReview:
    """Each of these reproduced a real defect before it was fixed.

    Kept as tests rather than fixed silently: every one is a case where a
    figure the data could not support was reading as a measurement, which is
    the failure this module exists to prevent and therefore the failure it is
    most likely to reintroduce.
    """

    def test_a_group_reviewed_but_almost_never_clean_yields_no_rate(self):
        """The measurability floor applies to the rate's own denominator.

        Counting reviewed donations let a group clear it on donations that were
        almost all confirmed risky, then produce a false-flag rate from the one
        clean observation — a tenfold disparity and a hard promotion failure
        from a single donation.
        """
        risky_heavy = [
            Cohort(
                donation_id=f"a-{i}",
                score=0.5,
                flagged=False,
                affiliation="A",
                reviewed=True,
                confirmed_risky=True,
            )
            for i in range(MIN_GROUP_CLEAN - 1)
        ] + [
            Cohort(
                donation_id="a-clean",
                score=0.5,
                flagged=True,
                affiliation="A",
                reviewed=True,
                confirmed_risky=False,
            )
        ]
        members = risky_heavy + cohorts(
            group="B", reviewed_clean=100, false_flags=10
        )
        report = differential_performance(members, attribute="affiliation")
        assert report.disparity is None
        assert report.within_tolerance is None

    def test_a_gap_the_intervals_do_not_separate_is_unresolved_not_clean(self):
        """At the measurability floor a 95% interval still spans a factor of
        two, so point estimates alone would fail a model on sampling noise. A
        gate that fails on noise is one people learn to override."""
        members = cohorts(group="A", reviewed_clean=30, false_flags=9) + cohorts(
            group="B", reviewed_clean=30, false_flags=5
        )
        report = differential_performance(members, attribute="affiliation")
        assert report.disparity > MAX_FALSE_FLAG_DISPARITY
        assert report.separated is False
        assert report.within_tolerance is None
        assert any("intervals overlap" in c for c in report.concerns())

    def test_a_gap_the_intervals_do_separate_is_a_finding(self):
        members = cohorts(group="A", reviewed_clean=400, false_flags=160) + cohorts(
            group="B", reviewed_clean=400, false_flags=40
        )
        report = differential_performance(members, attribute="affiliation")
        assert report.separated is True
        assert report.within_tolerance is False

    def test_widely_unknown_affiliation_blocks_rather_than_annotates(self):
        """A model passing a neutrality check on 5% of the population, with the
        caveat printed beside the word "pass", is the failure this whole module
        was written to prevent."""
        members = cohorts(group="PartaiA", reviewed_clean=200, false_flags=20) + cohorts(
            group="PartaiB", reviewed_clean=200, false_flags=20
        )
        # The unknowns form their own measurable group, at the same rate as
        # the known ones. Leaving them at zero false flags would make the
        # disparity unmeasurable for a different reason and test nothing.
        members += [
            Cohort(
                donation_id=f"unknown-{i}",
                score=0.4,
                flagged=i % 10 == 0,
                reviewed=True,
                confirmed_risky=False,
            )
            for i in range(8000)
        ]
        report = affiliation_assessment(members)
        assert report.errors.within_tolerance is True
        assert report.acceptable is None

    def test_a_measured_disparity_outranks_the_unknown_share_guard(self):
        """False and None both block, and they call for different responses.
        Downgrading a finding the data supports into "could not tell" loses the
        more informative answer."""
        members = cohorts(group="PartaiA", reviewed_clean=400, false_flags=160) + cohorts(
            group="PartaiB", reviewed_clean=400, false_flags=40
        )
        members += [
            Cohort(
                donation_id=f"u-{i}",
                score=0.4,
                flagged=i % 10 == 0,
                reviewed=True,
                confirmed_risky=False,
            )
            for i in range(8000)
        ]
        assert affiliation_assessment(members).acceptable is False

    def test_a_blank_affiliation_is_an_unknown_not_a_party(self):
        """`is not None` in one filter and truthiness in another let a blank
        count toward the total while contributing to no party, inflating every
        expected cell and manufacturing an association out of identical rates.
        """
        even = [
            Cohort(donation_id=f"a-{i}", score=0.5, flagged=i % 2 == 0, affiliation="A")
            for i in range(100)
        ] + [
            Cohort(donation_id=f"b-{i}", score=0.5, flagged=i % 2 == 0, affiliation="B")
            for i in range(100)
        ]
        assert cramers_v(even) == pytest.approx(0.0, abs=1e-9)

        blanks = [
            Cohort(donation_id=f"z-{i}", score=0.5, flagged=False, affiliation="")
            for i in range(100)
        ]
        assert cramers_v(even + blanks) == pytest.approx(0.0, abs=1e-9)
        assert affiliation_assessment(even + blanks).unknown_affiliation_share == (
            pytest.approx(1 / 3)
        )

    def test_a_negative_amount_is_refused_rather_than_banded(self):
        """The fallthrough put it in the band reserved for the largest
        contributions, so a sign error upstream would be reported there."""
        with pytest.raises(ValueError):
            size_band(-1)
