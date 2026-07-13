"""Topological Data Analysis: Vietoris-Rips persistent homology in Rust."""

from dataclasses import dataclass

from ._core import filtration_size
from ._core import persistent_homology as _persistent_homology
from ._core import persistent_homology_reps as _persistent_homology_reps


@dataclass(frozen=True)
class Representative:
    """Cocycle representative for one persistence interval.

    ``birth`` and ``death`` are simplices as vertex-id lists (``death`` is
    ``None`` for essential classes). ``cocycle`` is the representative cocycle
    of the class as a list of simplices (persistent cohomology, as in Ripser's
    ``do_cocycles``); in dimension 0 it is empty — there the birth vertex and
    the death edge carry the information.
    """

    birth: list
    death: list | None
    cocycle: list


@dataclass(frozen=True)
class Barcode:
    """Persistence diagrams plus aligned cocycle representatives.

    ``representatives[d][i]`` corresponds to ``diagrams[d][i]``.
    """

    diagrams: list
    representatives: list


def persistent_homology(
    data,
    max_dim=1,
    threshold=None,
    distance_matrix=False,
    quotient=False,
    peel=True,
    collapse=False,
    parallel=True,
    representatives=False,
):
    """Vietoris-Rips persistent homology.

    Returns a list of ``(k, 2)`` float32 arrays (one per dimension), or a
    :class:`Barcode` carrying diagrams and cocycle representatives when
    ``representatives=True``.
    """
    if not representatives:
        return _persistent_homology(
            data, max_dim, threshold, distance_matrix, quotient, peel, collapse, parallel
        )
    dgms, reps = _persistent_homology_reps(
        data, max_dim, threshold, distance_matrix, quotient, peel, collapse, parallel
    )
    return Barcode(
        diagrams=dgms,
        representatives=[
            [Representative(birth, death, cocycle) for (birth, death, cocycle) in dim]
            for dim in reps
        ],
    )


__all__ = [
    "Barcode",
    "Representative",
    "filtration_size",
    "persistent_homology",
]
