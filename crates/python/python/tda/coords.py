"""Circular and toroidal coordinates."""

from dataclasses import dataclass

from ._core import CircularMap
from ._core import circular_coordinates as _circular_coordinates
from ._core import circular_map as _circular_map


@dataclass(frozen=True)
class CircularCoordinates:
    """Angle columns with intervals, scales, and toroidal column groups."""

    coordinates: object
    intervals: object
    scales: object
    groups: list


def circular(
    data,
    n_classes=1,
    classes=None,
    scale=None,
    distance_matrix=False,
    quotient=False,
    prime=41,
    parallel=True,
):
    """Compute one circular-coordinate column per selected H1 class.

    ``n_classes`` selects the top-k classes by persistence; ``classes``
    overrides it with explicit ranks. Co-alive classes are jointly decorrelated
    into a toroidal block, reported by ``groups``.
    """
    coordinates, intervals, scales, group_ids = _circular_coordinates(
        data, n_classes, classes, scale, distance_matrix, quotient, prime, parallel
    )
    blocks = {}
    for column, group in enumerate(group_ids):
        blocks.setdefault(group, []).append(column)
    groups = [blocks[g] for g in sorted(blocks)]
    return CircularCoordinates(coordinates, intervals, scales, groups)


def circular_map(
    data,
    n_classes=1,
    classes=None,
    scale=None,
    prime=41,
    parallel=True,
):
    """Fit a circular map. Uncovered queries produce NaN values."""
    return _circular_map(data, n_classes, classes, scale, prime, parallel)


__all__ = [
    "CircularCoordinates",
    "CircularMap",
    "circular",
    "circular_map",
]
