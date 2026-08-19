"""Coordinates from persistent cohomology classes.

Maps data into the geometric target spaces classified by low-degree
cohomology: circular coordinates (H1 -> S1, jointly T^k for several classes),
real projective coordinates (H1 mod 2 -> RP^d), and complex projective
coordinates (H2 -> CP^d).

Coordinates are computed on the full complex: every data point is a vertex
and receives its value directly (no landmark subsampling). All functions
accept ``quotient=True`` to run on the quotient-cover filtration instead —
massively cheaper, log(3)-interleaved, blockier values (near-constant per
cluster), still one value per original point.
"""

from dataclasses import dataclass

import numpy as np

from ._core import CircularMap, ProjectiveMap
from ._core import circular_coordinates as _circular_coordinates
from ._core import circular_map as _circular_map
from ._core import complex_projective_coordinates as _complex_projective_coordinates
from ._core import complex_projective_map as _complex_projective_map
from ._core import projective_coordinates as _projective_coordinates
from ._core import projective_map as _projective_map


class ComplexProjectiveMap:
    """A fitted complex-projective map; ``transform`` evaluates the same
    CP^target_dim-valued map on new points (complex64 unit rows, up to a
    global phase; NaN rows outside the cover)."""

    def __init__(self, core):
        self._core = core

    @property
    def coordinates(self):
        return np.ascontiguousarray(self._core.coordinates).view(np.complex64)

    @property
    def interval(self):
        return self._core.interval

    @property
    def scale(self):
        return self._core.scale

    @property
    def target_dim(self):
        return self._core.target_dim

    def transform(self, data):
        return np.ascontiguousarray(self._core.transform(data)).view(np.complex64)


@dataclass(frozen=True)
class ComplexProjectiveCoordinates:
    """Complex projective coordinates for one H2 mod-p class.

    ``coordinates`` is an ``(n, target_dim + 1)`` complex64 array of unit
    rows, each defined up to a global phase (points of CP^target_dim; for
    ``target_dim=1``, CP^1 is the Riemann sphere). ``interval`` is the
    ``(birth, death)`` of the class used; ``scale`` the filtration value the
    map was computed at.
    """

    coordinates: object
    interval: tuple
    scale: float


@dataclass(frozen=True)
class CircularCoordinates:
    """Circular coordinates for k selected H1 classes.

    ``coordinates`` is an ``(n, k)`` float32 array of angles in ``[0, 2*pi)``,
    one column per class. ``intervals`` is ``(k, 2)`` of ``[birth, death]``,
    ``scales`` the filtration value each column was computed at, and
    ``groups`` the toroidal blocks: columns listed together were jointly
    decorrelated (single-column groups are plain circular coordinates).
    """

    coordinates: object
    intervals: object
    scales: object
    groups: list


@dataclass(frozen=True)
class ProjectiveCoordinates:
    """Real projective coordinates for one H1 mod-2 class.

    ``coordinates`` is an ``(n, target_dim + 1)`` float32 array of unit rows,
    each defined up to sign (points of RP^target_dim). ``interval`` is the
    ``(birth, death)`` of the class used; ``scale`` is the filtration value
    the map was computed at.
    """

    coordinates: object
    interval: tuple
    scale: float


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
    """Circular coordinates from H1 classes (one angle column per class).

    ``n_classes`` selects the top-k classes by persistence; ``classes``
    overrides it with explicit persistence ranks. ``scale`` defaults to the
    midpoint of each class interval (an explicit value keeps only the classes
    alive at it). ``prime`` is the coefficient field behind the integer lift
    (odd prime). Co-alive classes are jointly harmonised and
    lattice-decorrelated (toroidal coordinates); ``groups`` names the blocks.
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
    """Fit a circular-coordinates map; evaluate it on new data with
    ``.transform(new_points)``.

    Same selection knobs as :func:`circular`. The fitted :class:`CircularMap`
    keeps the training points, harmonic potentials, and integer cocycles in
    Rust; ``transform`` evaluates the same circle-valued map on new points
    and returns NaN for points outside the cover (farther than half the
    class scale from every training point). ``distance_matrix`` and
    ``quotient`` are not supported for fitted maps.
    """
    return _circular_map(data, n_classes, classes, scale, prime, parallel)


def projective_map(
    data,
    class_index=0,
    target_dim=2,
    scale=None,
    parallel=True,
):
    """Fit a real-projective map; evaluate it on new data with
    ``.transform(new_points)`` (unit rows up to sign; NaN rows outside the
    cover). ``distance_matrix`` and ``quotient`` are not supported for
    fitted maps."""
    return _projective_map(data, class_index, target_dim, scale, parallel)


def complex_projective_map(
    data,
    class_index=0,
    target_dim=1,
    scale=None,
    prime=41,
    parallel=True,
):
    """Fit a complex-projective map; evaluate it on new data with
    ``.transform(new_points)`` (complex64 unit rows up to a global phase;
    NaN rows outside the cover). ``distance_matrix`` and ``quotient`` are
    not supported for fitted maps."""
    core = _complex_projective_map(data, class_index, target_dim, scale, prime, parallel)
    return ComplexProjectiveMap(core)


def complex_projective(
    data,
    class_index=0,
    target_dim=1,
    scale=None,
    distance_matrix=False,
    quotient=False,
    prime=41,
    parallel=True,
):
    """Complex projective coordinates from an H2 mod-p class.

    ``class_index`` selects the class by persistence rank. The default
    ``target_dim=1`` maps into CP^1 (the Riemann sphere). ``prime`` is the
    coefficient field behind the integer lift (odd prime).
    """
    coordinates, birth, death, used_scale = _complex_projective_coordinates(
        data, class_index, target_dim, scale, distance_matrix, quotient, prime, parallel
    )
    complex_view = np.ascontiguousarray(coordinates).view(np.complex64)
    return ComplexProjectiveCoordinates(complex_view, (birth, death), used_scale)


def projective(
    data,
    class_index=0,
    target_dim=2,
    scale=None,
    distance_matrix=False,
    quotient=False,
    parallel=True,
):
    """Real projective coordinates from an H1 mod-2 class.

    ``class_index`` selects the class by persistence rank (0 = most
    persistent). ``scale`` defaults to the midpoint of the class interval and
    must otherwise lie inside ``[birth, death)``. ``data`` is an ``(n, d)``
    point cloud or an ``(n, n)`` distance matrix (``distance_matrix=True``);
    sparse input is not supported.
    """
    coordinates, birth, death, used_scale = _projective_coordinates(
        data, class_index, target_dim, scale, distance_matrix, quotient, parallel
    )
    return ProjectiveCoordinates(coordinates, (birth, death), used_scale)


__all__ = [
    "CircularCoordinates",
    "CircularMap",
    "ComplexProjectiveCoordinates",
    "ComplexProjectiveMap",
    "ProjectiveCoordinates",
    "ProjectiveMap",
    "circular",
    "circular_map",
    "complex_projective",
    "complex_projective_map",
    "projective",
    "projective_map",
]
