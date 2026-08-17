"""Whether two features are the same feature.

The previous feature set shipped two columns labelled as graph centrality that
were verbatim copies of two counting columns. Nothing caught it, because every
check in place asked whether the names were distinct, and they were. A model
trained on that set splits its attributed importance across the copies, so the
one thing that would have made the duplication visible — an importance ranking
— is the thing the duplication corrupts.

Three kinds of redundancy are detected, and they are not equally severe.

**Identical.** Two columns hold the same value in every row. One of them is
dead weight and the pair should not both exist.

**Affine.** One column is the other rescaled: ``y = a·x + b`` exactly. A tree
model is invariant to monotone rescaling, so an affine copy carries no
information the original lacks while still splitting importance. Detected by
solving for the coefficients on two rows and verifying every remaining row,
rather than by thresholding a correlation — a threshold invites tuning, and the
question here has an exact answer.

**Constant.** A column that never varies cannot inform a split. Reported
separately because the cause is usually different: a feature whose inputs are
absent in this dataset, which is a data problem rather than a definition one.

Everything is reported over the rows actually supplied. On a handful of rows
distinct columns collide by chance, so below a floor the check reports that it
could not run rather than a clean result — a redundancy check that passes
vacuously is worse than none, because it is the one people cite.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

FeatureValue = float | int | bool | str | None

#: Below this many rows, two distinct features can hold the same values by
#: coincidence often enough that a match means nothing.
MIN_ROWS = 50

#: Present observations a pair needs before an affine fit means anything. A fit
#: verified on the same points that produced it is not a verification.
MIN_PAIRS = 20

#: Tolerance for the affine check. Tight enough that only an actual rescaling
#: passes; loose enough to survive floating-point accumulation.
AFFINE_TOLERANCE = 1e-9

Kind = Literal["identical", "affine", "constant"]


@dataclass(frozen=True)
class Redundancy:
    """One finding."""

    kind: Kind
    columns: tuple[str, ...]
    detail: str

    def describe(self) -> str:
        return f"{self.kind}: {', '.join(self.columns)} — {self.detail}"


@dataclass(frozen=True)
class RedundancyReport:
    rows: int
    findings: tuple[Redundancy, ...]
    unmeasurable_reason: str | None = None

    @property
    def clean(self) -> bool | None:
        """None when the check could not run.

        Not True. A check that could not run has found nothing, which is not
        the same as there being nothing to find.
        """
        if self.unmeasurable_reason is not None:
            return None
        return not self.findings

    def of_kind(self, kind: Kind) -> tuple[Redundancy, ...]:
        return tuple(f for f in self.findings if f.kind == kind)

    def describe(self) -> str:
        if self.unmeasurable_reason:
            return f"redundancy not measurable — {self.unmeasurable_reason}"
        if not self.findings:
            return f"no redundant columns over {self.rows} rows"
        lines = [f"{len(self.findings)} redundancy finding(s) over {self.rows} rows:"]
        lines.extend(f"  {finding.describe()}" for finding in self.findings)
        return "\n".join(lines)


def _columns(
    rows: Sequence[Mapping[str, FeatureValue]],
) -> dict[str, list[FeatureValue]]:
    names: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for name in row:
            if name not in seen:
                seen.add(name)
                names.append(name)
    return {name: [row.get(name) for row in rows] for name in names}


def _is_numeric(values: Sequence[FeatureValue]) -> bool:
    return any(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)


def _affine(
    left: Sequence[FeatureValue], right: Sequence[FeatureValue]
) -> tuple[float, float] | None:
    """Solve ``right = a·left + b`` and verify it, or return None.

    Nulls must line up: a column that is null where another has a value is
    describing something different, however well the present values fit.
    """
    pairs: list[tuple[float, float]] = []
    for x, y in zip(left, right):
        if x is None or y is None:
            if x is not None or y is not None:
                return None
            continue
        if isinstance(x, bool) or isinstance(y, bool):
            return None
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            return None
        pairs.append((float(x), float(y)))

    # Two points always determine a line, and the verification loop would then
    # iterate over those same two points. Two mostly-null columns holding
    # unrelated values in the same two rows were reported as a rescaled copy.
    if len(pairs) < MIN_PAIRS:
        return None

    base_x, base_y = pairs[0]
    anchor = next((p for p in pairs[1:] if p[0] != base_x), None)
    if anchor is None:
        # left is constant over the present rows; the constant check covers it,
        # and any slope would fit.
        return None

    slope = (anchor[1] - base_y) / (anchor[0] - base_x)
    if slope == 0.0:
        return None
    intercept = base_y - slope * base_x

    scale = max(abs(y) for _, y in pairs) or 1.0
    for x, y in pairs:
        if not math.isclose(
            slope * x + intercept, y, rel_tol=AFFINE_TOLERANCE, abs_tol=scale * AFFINE_TOLERANCE
        ):
            return None
    return (slope, intercept)


def detect_redundancy(
    rows: Sequence[Mapping[str, FeatureValue]],
) -> RedundancyReport:
    """Find columns that duplicate one another.

    Takes computed feature rows rather than the definitions, because the defect
    is in the values. Two definitions can look entirely different and compute
    the same thing, which is precisely how the previous set acquired its copies.
    """
    if len(rows) < MIN_ROWS:
        return RedundancyReport(
            rows=len(rows),
            findings=(),
            unmeasurable_reason=(
                f"{len(rows)} rows, below the {MIN_ROWS} at which a value match "
                f"means more than coincidence"
            ),
        )

    columns = _columns(rows)
    findings: list[Redundancy] = []

    constant: set[str] = set()
    for name, values in columns.items():
        distinct = {v for v in values}
        if len(distinct) <= 1:
            constant.add(name)
            only = next(iter(distinct), None)
            findings.append(
                Redundancy(
                    kind="constant",
                    columns=(name,),
                    detail=(
                        f"every row holds {only!r}; the feature cannot inform a "
                        f"split on this data"
                    ),
                )
            )

    varying = {n: v for n, v in columns.items() if n not in constant}

    # Exact duplicates first: group by the value sequence so the comparison is
    # linear rather than quadratic in the number of columns.
    groups: dict[tuple, list[str]] = defaultdict(list)
    for name, values in varying.items():
        groups[tuple((type(v).__name__, v) for v in values)].append(name)
    # Only the non-representative members are withheld from the affine pass.
    # Dropping a whole identical group hid a rescaled copy of it: with a == b
    # and c == 2a, both a and b left the numeric set and c had nothing to be
    # compared against — the exact defect this module exists to catch, escaping
    # because a different pair was caught first.
    duplicated: set[str] = set()
    for members in groups.values():
        if len(members) > 1:
            ordered = tuple(sorted(members))
            duplicated.update(ordered[1:])
            findings.append(
                Redundancy(
                    kind="identical",
                    columns=ordered,
                    detail=(
                        "identical in every row; a model splits its attributed "
                        "importance between them and neither reads as important"
                    ),
                )
            )

    # Affine copies among what is left. Quadratic, but over a few dozen numeric
    # columns, and it runs once per training run.
    numeric = sorted(
        name
        for name, values in varying.items()
        if name not in duplicated and _is_numeric(values)
    )
    for index, left in enumerate(numeric):
        for right in numeric[index + 1 :]:
            solved = _affine(columns[left], columns[right])
            if solved is None:
                continue
            slope, intercept = solved
            findings.append(
                Redundancy(
                    kind="affine",
                    columns=(left, right),
                    detail=(
                        f"{right} = {slope:g}·{left} + {intercept:g} in every row; "
                        f"a rescaled copy carries nothing the original lacks"
                    ),
                )
            )

    return RedundancyReport(rows=len(rows), findings=tuple(findings))
