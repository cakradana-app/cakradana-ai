"""The dependency-pinning gate.

A CI check with no tests of its own fails open: it reports success, nobody
looks, and the thing it was meant to catch passes through. Both defects below
did exactly that — the script ran green while silently examining a fraction of
what it claimed to check.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from check_pins import declared, normalise, pinned, satisfies, version_tuple  # noqa: E402


class TestParsingTheManifest:
    def test_an_extras_spec_does_not_truncate_the_array(self):
        """A non-greedy match stopped at the first "]", so everything declared
        after an extras spec vanished — and the check reported success on the
        remainder."""
        pyproject = """
dependencies = [
    "uvicorn[standard]>=0.30,<1.0",
    "pyyaml>=6.0,<7.0",
    "lightgbm>=4.3,<5.0",
]
"""
        assert set(declared(pyproject, "dependencies")) == {
            "uvicorn",
            "pyyaml",
            "lightgbm",
        }

    def test_the_name_is_separated_from_its_constraint(self):
        found = declared('dependencies = [\n    "scikit-learn>=1.5,<2.0",\n]\n', "dependencies")
        assert found == {"scikit-learn": "scikit-learn>=1.5,<2.0"}

    def test_a_missing_section_is_empty_rather_than_an_error(self):
        assert declared("", "dependencies") == {}

    def test_names_are_compared_in_a_normalised_form(self):
        assert normalise("scikit_learn") == normalise("scikit-learn")


class TestReadingTheRequirements:
    def test_an_exact_pin_is_read(self):
        assert pinned("lightgbm==4.3.0\n") == {"lightgbm": "4.3.0"}

    def test_a_comment_and_an_include_are_skipped(self):
        text = "-r requirements.txt\n# a note\npytest==8.2.2\n"
        assert pinned(text) == {"pytest": "8.2.2"}

    def test_a_requirement_without_a_pin_is_recorded_as_unpinned(self):
        """Recorded rather than dropped: "present but not pinned" and "absent"
        are different problems and get different messages."""
        assert pinned("fastapi>=0.111\n") == {"fastapi": ""}


class TestConstraints:
    def test_a_version_inside_the_range_satisfies_it(self):
        assert satisfies("4.3.0", "lightgbm>=4.3,<5.0") is None

    def test_a_version_below_the_floor_does_not(self):
        assert "does not satisfy >=4.3" in (satisfies("3.9.0", "lightgbm>=4.3,<5.0") or "")

    def test_a_version_at_the_ceiling_does_not(self):
        assert satisfies("5.0.0", "lightgbm>=4.3,<5.0") is not None

    def test_a_compatible_release_constraint_is_checked(self):
        """`~=` was absent from the operator alternation, so a compatible-release
        constraint was read as no constraint at all."""
        assert satisfies("1.4.5", "joblib~=1.4.2") is None
        assert satisfies("1.5.0", "joblib~=1.4.2") is not None
        assert satisfies("1.4.1", "joblib~=1.4.2") is not None

    def test_the_two_character_operators_are_not_split(self):
        """A leading `<` in the alternation would match the tail of `<=` and
        read the wrong bound."""
        assert satisfies("2.0.0", "x<=2.0") is None
        assert satisfies("2.0.1", "x<=2.0") is not None

    def test_a_prerelease_pin_is_refused_rather_than_compared(self):
        """Numeric components alone would compare a release candidate as its
        release, which is a different version."""
        assert "not a plain release version" in (satisfies("2.0.0rc1", "x>=1.0") or "")

    def test_components_of_different_lengths_compare(self):
        assert version_tuple("1.26") < version_tuple("1.26.4")
        assert satisfies("1.26.4", "numpy>=1.26,<3.0") is None


class TestTheRealManifests:
    def test_this_project_passes_its_own_check(self):
        from check_pins import check

        assert check("dependencies", "requirements.txt") == []
