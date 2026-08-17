"""Duplicate-feature detection.

The check that matters is the last class: the real catalogue, computed over a
real population, asserted to contain no column that is a copy of another. The
unit tests above it exist so that a failure there can be read.
"""

from __future__ import annotations

import pytest

from cakradana.data import GeneratorConfig, generate
from cakradana.features import FeatureService, detect_redundancy
from cakradana.features.redundancy import MIN_ROWS
from cakradana.history import InMemoryDonationStore
from cakradana.rules import load_latest

SMALL = GeneratorConfig(
    seed=41,
    n_legitimate_donors=200,
    n_recipients=6,
    n_background_donations=900,
    n_grassroots_campaigns=3,
)


def rows(count: int = MIN_ROWS, **columns) -> list[dict]:
    return [
        {name: values[index % len(values)] for name, values in columns.items()}
        for index in range(count)
    ]


class TestMeasurability:
    def test_too_few_rows_is_not_a_clean_result(self):
        report = detect_redundancy(rows(10, a=[1, 2], b=[1, 2]))
        assert report.clean is None
        assert report.findings == ()
        assert "below the" in (report.unmeasurable_reason or "")

    def test_at_the_floor_the_check_runs(self):
        report = detect_redundancy(rows(MIN_ROWS, a=[1, 2, 3], b=[3, 1, 2]))
        assert report.clean is True


class TestIdentical:
    def test_two_columns_with_the_same_values_are_reported(self):
        report = detect_redundancy(rows(a=[1, 2, 3], b=[1, 2, 3]))
        finding = report.of_kind("identical")[0]
        assert finding.columns == ("a", "b")
        assert "importance" in finding.detail

    def test_three_copies_report_as_one_group(self):
        report = detect_redundancy(rows(a=[1, 2, 3], b=[1, 2, 3], c=[1, 2, 3]))
        assert len(report.of_kind("identical")) == 1
        assert report.of_kind("identical")[0].columns == ("a", "b", "c")

    def test_distinct_columns_are_not_reported(self):
        assert detect_redundancy(rows(a=[1, 2, 3], b=[3, 1, 2])).clean is True

    def test_a_matching_int_and_bool_are_not_the_same_column(self):
        """True == 1 in Python. Two features whose declared types differ are
        describing different things whatever the comparison says."""
        report = detect_redundancy(rows(a=[1, 0], b=[True, False]))
        assert report.of_kind("identical") == ()

    def test_columns_that_agree_only_on_their_nulls_are_still_identical(self):
        report = detect_redundancy(rows(a=[None, 2], b=[None, 2]))
        assert report.of_kind("identical")

    def test_a_null_where_the_other_has_a_value_is_a_difference(self):
        report = detect_redundancy(rows(a=[None, 2, 3], b=[1, 2, 3]))
        assert report.of_kind("identical") == ()


class TestAffine:
    def test_a_rescaled_copy_is_reported(self):
        report = detect_redundancy(rows(a=[1, 2, 3, 4], b=[2, 4, 6, 8]))
        finding = report.of_kind("affine")[0]
        assert finding.columns == ("a", "b")
        assert "rescaled copy" in finding.detail

    def test_a_shifted_copy_is_reported(self):
        report = detect_redundancy(rows(a=[1, 2, 3, 4], b=[11, 12, 13, 14]))
        assert report.of_kind("affine")

    def test_a_column_that_only_nearly_fits_is_not_a_copy(self):
        report = detect_redundancy(rows(a=[1, 2, 3, 4], b=[2, 4, 6, 8.5]))
        assert report.of_kind("affine") == ()

    def test_the_check_is_exact_rather_than_a_correlation_threshold(self):
        """A threshold would have to be chosen, and a chosen threshold gets
        moved when it is inconvenient."""
        report = detect_redundancy(rows(a=[1, 2, 3, 4], b=[1.0000001, 2, 3, 4]))
        assert report.of_kind("affine") == ()

    def test_misaligned_nulls_defeat_the_fit(self):
        report = detect_redundancy(rows(a=[1, 2, None, 4], b=[2, 4, 6, 8]))
        assert report.of_kind("affine") == ()

    def test_a_negated_copy_is_still_a_copy(self):
        report = detect_redundancy(rows(a=[1, 2, 3, 4], b=[-1, -2, -3, -4]))
        assert report.of_kind("affine")

    def test_categorical_columns_are_not_fitted(self):
        report = detect_redundancy(rows(a=["x", "y", "z"], b=["p", "q", "r"]))
        assert report.of_kind("affine") == ()


class TestConstant:
    def test_a_column_that_never_varies_is_reported(self):
        report = detect_redundancy(rows(a=[7], b=[1, 2, 3]))
        finding = report.of_kind("constant")[0]
        assert finding.columns == ("a",)
        assert "cannot inform a split" in finding.detail

    def test_an_all_null_column_is_constant(self):
        assert detect_redundancy(rows(a=[None], b=[1, 2, 3])).of_kind("constant")

    def test_two_constant_columns_are_not_also_reported_as_duplicates(self):
        """Reporting the same defect twice under two names makes a report
        longer without making it more informative."""
        report = detect_redundancy(rows(a=[7], b=[7], c=[1, 2, 3]))
        assert len(report.of_kind("constant")) == 2
        assert report.of_kind("identical") == ()


@pytest.fixture(scope="module")
def matrix() -> list[dict]:
    """The real catalogue computed over a generated population.

    The generated dataset is used rather than a hand-built one because the
    question is whether two features coincide in general, and a small
    hand-built graph coincides for reasons of its own: make every donor reach
    every recipient and shared-counterparty count becomes in-degree minus one
    by construction, with nothing wrong with either feature.
    """
    dataset = generate(SMALL)
    service = FeatureService(
        load_latest(), calendar=dataset.calendar, registers=dataset.registers
    )
    store = InMemoryDonationStore(dataset.donations)
    return [
        vector.values
        for _, vector in service.backfill(store, entities=dataset.entities)
    ]


class TestTheRealCatalogue:
    """MR-93 as it actually binds: no shipped feature duplicates another."""

    def test_the_population_is_large_enough_to_measure(self, matrix):
        assert len(matrix) >= MIN_ROWS

    def test_no_shipped_feature_duplicates_another(self, matrix):
        report = detect_redundancy(matrix)
        identical = report.of_kind("identical")
        affine = report.of_kind("affine")
        assert not identical and not affine, report.describe()

    def test_only_the_declared_features_are_constant(self, matrix):
        """A constant column is usually a data problem rather than a definition
        one, and on generated data these two are expected: the generator
        resolves every entity perfectly, so resolution confidence is always 1.0
        and nothing is ever unresolved. Any *other* constant means a feature
        stopped computing, which the fixed list is here to surface."""
        expected = {"entity_resolution_confidence", "has_unresolved_entity"}
        found = {
            finding.columns[0]
            for finding in detect_redundancy(matrix).of_kind("constant")
        }
        assert found == expected

    def test_the_previously_duplicated_columns_are_still_absent(self, matrix):
        """The defect this check exists for: two columns labelled as graph
        centrality that were verbatim copies of two counting columns."""
        assert "degree_centrality_sender" not in matrix[0]
        assert "degree_centrality_receiver" not in matrix[0]
