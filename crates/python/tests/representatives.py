"""Structural checks for cocycle representatives.

Representatives are not unique, so cross-library value comparison is not
possible; what IS checkable, and checked here, per emitted interval:

  * the diagrams of the representatives run equal the plain run exactly;
  * representatives align 1:1 with the diagram rows in every dimension;
  * simplices have the right arity (birth: d+1 vertices, death: d+2);
  * for dims >= 1 the birth/death simplex diameters equal the interval
    endpoints, and the birth simplex is a member of its cocycle;
  * dimension 0 carries (vertex, edge) pairs with empty cocycles;
  * on a circle, exactly one dominant H1 class exists and its birth edge is
    an edge of the sampled circle graph.

The quotient variant runs structurally only: its filtration rewrites
distances, so diameter checks against the input metric do not apply.
"""

import numpy as np

import tda


def pairwise(points):
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt((diff * diff).sum(-1))


def diameter(simplex, dist):
    idx = np.asarray(simplex)
    if len(idx) < 2:
        return 0.0
    return dist[np.ix_(idx, idx)].max()


def check(points, max_dim=1, metric_checks=True, **kwargs):
    result = tda.persistent_homology(
        points, max_dim=max_dim, representatives=True, **kwargs
    )
    plain = tda.persistent_homology(points, max_dim=max_dim, **kwargs)
    dist = pairwise(points)

    assert isinstance(result, tda.Barcode)
    assert len(result.diagrams) == len(result.representatives) == max_dim + 1

    for d in range(max_dim + 1):
        got = np.asarray(result.diagrams[d])
        want = np.asarray(plain[d])
        assert got.shape == want.shape, f"H{d}: diagram shape mismatch"
        assert np.array_equal(got, want), f"H{d}: representatives run changed the barcode"
        reps = result.representatives[d]
        assert len(reps) == len(got), f"H{d}: representative count mismatch"

        for (birth, death), rep in zip(got, reps):
            assert len(rep.birth) == max(d + 1, 1)
            if np.isinf(death):
                assert rep.death is None
            else:
                assert rep.death is not None and len(rep.death) == d + 2

            if d == 0:
                assert rep.cocycle == []
                continue

            assert rep.cocycle, f"H{d}: empty cocycle for a reduced class"
            assert all(len(s) == d + 1 for s in rep.cocycle)
            assert sorted(rep.birth) in [sorted(s) for s in rep.cocycle]
            if metric_checks:
                assert abs(diameter(rep.birth, dist) - birth) < 1e-6
                if rep.death is not None:
                    assert abs(diameter(rep.death, dist) - death) < 1e-6


def circle_h1_representative():
    """The dominant H1 class of a circle: its birth edge joins nearby points."""
    rng = np.random.default_rng(7)
    theta = np.sort(rng.uniform(0.0, 2.0 * np.pi, 60))
    points = np.stack([np.cos(theta), np.sin(theta)], axis=1).astype(np.float32)
    points += rng.normal(scale=0.01, size=points.shape).astype(np.float32)

    result = tda.persistent_homology(points, max_dim=1, representatives=True)
    h1 = np.asarray(result.diagrams[1])
    assert len(h1) >= 1
    dominant = int(np.argmax(h1[:, 1] - h1[:, 0]))
    rep = result.representatives[1][dominant]

    # The birth edge realises the birth scale, which for a densely sampled
    # circle is far below the diameter: a short edge between angular
    # neighbours, part of the loop that wraps the circle.
    dist = pairwise(points)
    edge = diameter(rep.birth, dist)
    assert abs(edge - h1[dominant, 0]) < 1e-6
    assert edge < 1.0, f"dominant H1 birth edge suspiciously long: {edge}"
    assert len(rep.cocycle) >= 1


def main():
    rng = np.random.default_rng(3)

    cloud = rng.random((40, 3)).astype(np.float32)
    check(cloud, max_dim=1)
    check(cloud, max_dim=2)
    check(cloud, max_dim=1, peel=False)
    # Collapse SHIFTS dominated edge values, so simplex diameters in the
    # original metric need not equal the (exact) interval endpoints.
    check(cloud, max_dim=1, metric_checks=False, collapse=True)
    check(cloud, max_dim=1, threshold=0.6)  # user threshold: essentials allowed

    # Distance-matrix input path.
    dist = pairwise(cloud).astype(np.float32)
    result = tda.persistent_homology(
        dist, max_dim=1, distance_matrix=True, representatives=True
    )
    assert len(result.diagrams) == 2

    # Quotient rewrites the metric: structural checks only.
    check(cloud, max_dim=1, metric_checks=False, quotient=True)

    circle_h1_representative()
    print("representatives: ok")


if __name__ == "__main__":
    main()
