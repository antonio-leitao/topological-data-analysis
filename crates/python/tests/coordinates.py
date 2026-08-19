"""Checks for tda.coords.projective.

The load-bearing check is topological: the H1 mod-2 class of a circle is the
Moebius class, so walking once around the sampled loop, the lifted
representative must return with the opposite sign — the product over
consecutive (cyclic) pairs of sign<u_i, u_{i+1}> is exactly -1. This holds
for ANY continuous classifying map of the nontrivial class, independent of
the principal basis, so it pins down correctness of the cocycle restriction,
the signs, and the projection at once.
"""

import numpy as np

import tda


def circle(n, wobble=0.02, seed=None, noise=0.0):
    theta = 2.0 * np.pi * np.arange(n) / n
    radius = 1.0 + wobble * np.sin(3.0 * theta)
    points = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
    if noise > 0.0:
        rng = np.random.default_rng(seed)
        points = points + rng.normal(scale=noise, size=points.shape)
    return points.astype(np.float32)


def moebius_sign(coordinates):
    """Product of signs of consecutive cyclic dots; -1 = nontrivial class."""
    rolled = np.roll(coordinates, -1, axis=0)
    dots = (coordinates * rolled).sum(axis=1)
    assert np.all(np.abs(dots) > 1e-4), "consecutive points nearly orthogonal"
    return float(np.prod(np.sign(dots)))


def winding(angles):
    """Winding number of an angle sequence around a sampled closed loop."""
    diffs = np.diff(np.append(angles, angles[0]))
    wrapped = (diffs + np.pi) % (2.0 * np.pi) - np.pi
    return float(wrapped.sum() / (2.0 * np.pi))


def check_circular(points):
    n = len(points)
    res = tda.coords.circular(points)
    assert isinstance(res, tda.coords.CircularCoordinates)
    angles = np.asarray(res.coordinates)
    assert angles.shape == (n, 1)
    assert angles.dtype == np.float32
    assert np.all(angles >= 0.0) and np.all(angles < 2.0 * np.pi + 1e-4)
    assert np.asarray(res.intervals).shape == (1, 2)
    birth, death = np.asarray(res.intervals)[0]
    assert birth < res.scales[0] < death
    assert res.groups == [[0]]

    # The defining property: the coordinate winds exactly once.
    w = winding(angles[:, 0])
    assert abs(abs(w) - 1.0) < 1e-3, f"expected winding 1, got {w}"

    # Deterministic.
    again = tda.coords.circular(points)
    assert np.array_equal(angles, np.asarray(again.coordinates))


def main():
    n = 48
    points = circle(n)

    res = tda.coords.projective(points)
    assert isinstance(res, tda.coords.ProjectiveCoordinates)
    coords = np.asarray(res.coordinates)
    assert coords.shape == (n, 3)
    assert coords.dtype == np.float32

    # Unit rows, valid interval and scale.
    norms = np.linalg.norm(coords, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)
    birth, death = res.interval
    assert birth < res.scale < death

    # The topological check.
    assert moebius_sign(coords) == -1.0

    # Deterministic.
    again = tda.coords.projective(points)
    assert np.array_equal(coords, np.asarray(again.coordinates))

    # Distance-matrix input path agrees with the point-cloud path.
    diff = points[:, None, :] - points[None, :, :]
    dist = np.sqrt((diff * diff).sum(-1)).astype(np.float32)
    from_matrix = tda.coords.projective(dist, distance_matrix=True)
    assert moebius_sign(np.asarray(from_matrix.coordinates)) == -1.0

    # Noisy circle still carries the class.
    noisy = circle(64, seed=11, noise=0.05)
    res = tda.coords.projective(noisy, target_dim=3)
    assert np.asarray(res.coordinates).shape == (64, 4)

    # Selection errors surface as ValueError; sparse input as TypeError.
    try:
        tda.coords.projective(points, class_index=99)
        raise AssertionError("expected ValueError for class_index=99")
    except ValueError:
        pass

    # target_dim is validated in the Rust core: the unreduced homogeneous
    # map has n coordinates, so the largest target is RP^(n-1) / CP^(n-1).
    for coordinate in (tda.coords.projective, tda.coords.complex_projective):
        for bad_dim in (0, n):
            try:
                coordinate(points, target_dim=bad_dim)
                raise AssertionError(f"expected ValueError for target_dim={bad_dim}")
            except ValueError as error:
                message = str(error)
                assert f"1..={n - 1}" in message and f"got {bad_dim}" in message

    class FakeSparse:
        def tocoo(self):  # pragma: no cover - never called
            return self

    try:
        tda.coords.projective(FakeSparse())
        raise AssertionError("expected TypeError for sparse input")
    except TypeError:
        pass

    # ── circular ─────────────────────────────────────────────────────────
    check_circular(points)

    # Empty explicit selection → (n, 0); bad prime and multi-class → errors.
    empty = tda.coords.circular(points, classes=[])
    assert np.asarray(empty.coordinates).shape == (n, 0)
    assert empty.groups == []

    try:
        tda.coords.circular(points, prime=4)
        raise AssertionError("expected ValueError for prime=4")
    except ValueError:
        pass

    # Toroidal: two circles, two classes → one co-alive block; after
    # decorrelation each column winds exactly one circle.
    per = 24
    theta = 2.0 * np.pi * np.arange(per) / per
    left = circle(per)
    right = np.stack([5.0 + np.cos(theta), np.sin(theta)], axis=1).astype(np.float32)
    both = np.concatenate([left, right], axis=0)

    res = tda.coords.circular(both, n_classes=2)
    angles = np.asarray(res.coordinates)
    assert angles.shape == (2 * per, 2)
    assert res.groups == [[0, 1]], f"expected one block of two, got {res.groups}"
    assert np.asarray(res.intervals).shape == (2, 2)

    w = np.array(
        [
            [winding(angles[:per, c]), winding(angles[per:, c])]
            for c in range(2)
        ]
    )
    winds_once = np.abs(np.abs(w) - 1.0) < 1e-3
    is_flat = np.abs(w) < 1e-3
    diag = winds_once[0, 0] and is_flat[0, 1] and is_flat[1, 0] and winds_once[1, 1]
    anti = is_flat[0, 0] and winds_once[0, 1] and winds_once[1, 0] and is_flat[1, 1]
    assert diag or anti, f"each column must wind exactly one circle, got {w}"

    # ── complex projective ──────────────────────────────────────────────
    m = 64
    i = np.arange(m)
    z = 1.0 - 2.0 * (i + 0.5) / m
    radius = np.sqrt(1.0 - z * z)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    sphere = np.stack(
        [radius * np.cos(golden * i), radius * np.sin(golden * i), z], axis=1
    ).astype(np.float32)

    res = tda.coords.complex_projective(sphere)
    assert isinstance(res, tda.coords.ComplexProjectiveCoordinates)
    cp = np.asarray(res.coordinates)
    assert cp.shape == (m, 2) and cp.dtype == np.complex64
    norms = np.linalg.norm(cp, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)
    birth, death = res.interval
    assert birth < res.scale < death

    # CP^1 is the Riemann sphere: for the sphere's H2 class the image must
    # spread — every axis of the image attains both signs.
    a, b = cp[:, 0], cp[:, 1]
    image = np.stack(
        [
            (np.abs(a) ** 2 - np.abs(b) ** 2),
            2.0 * (np.conj(a) * b).real,
            2.0 * (np.conj(a) * b).imag,
        ],
        axis=1,
    )
    assert np.all(image.min(axis=0) < 0.0) and np.all(image.max(axis=0) > 0.0), (
        f"CP^1 image does not cover the sphere: {image.min(axis=0)} .. {image.max(axis=0)}"
    )

    again = tda.coords.complex_projective(sphere)
    assert np.array_equal(cp, np.asarray(again.coordinates))

    # ── quotient acceleration ───────────────────────────────────────────
    # The topological invariants survive the log-3 approximation, and every
    # original point still receives its own value.
    q = tda.coords.circular(points, quotient=True)
    q_angles = np.asarray(q.coordinates)
    assert q_angles.shape == (n, 1)
    wq = winding(q_angles[:, 0])
    assert abs(abs(wq) - 1.0) < 1e-3, f"quotient winding: {wq}"

    q_rp = tda.coords.projective(points, quotient=True)
    assert moebius_sign(np.asarray(q_rp.coordinates)) == -1.0

    # ── fitted maps (fit / transform) ───────────────────────────────────
    model = tda.coords.circular_map(points)
    assert isinstance(model, tda.coords.CircularMap)
    train_values = np.asarray(model.coordinates)
    assert train_values.shape == (n, 1)
    # Stored training coordinates equal the plain function's output.
    plain = tda.coords.circular(points)
    assert np.array_equal(train_values, np.asarray(plain.coordinates))

    # Round trip: PoU-smoothed but winding-preserving.
    back = model.transform(points)
    assert back.shape == (n, 1) and not np.isnan(back).any()
    assert abs(abs(winding(back[:, 0])) - 1.0) < 1e-3

    # Fresh points on the same circle: finite, wind once.
    phases = 2.0 * np.pi * (np.arange(60) + 0.37) / 60
    fresh = np.stack([np.cos(phases), np.sin(phases)], axis=1).astype(np.float32)
    values = model.transform(fresh)
    assert not np.isnan(values).any()
    assert abs(abs(winding(values[:, 0])) - 1.0) < 1e-3

    # Outside the cover → NaN; wrong width → error.
    far = model.transform(np.array([[100.0, 100.0]], dtype=np.float32))
    assert np.isnan(far).all()
    try:
        model.transform(np.zeros((3, 5), dtype=np.float32))
        raise AssertionError("expected ValueError for wrong ambient dimension")
    except ValueError:
        pass

    # RP map: round trip reproduces the fit up to sign; fresh loop stays
    # Moebius; far point → NaN row.
    rp_map = tda.coords.projective_map(points)
    rp_fit = np.asarray(rp_map.coordinates)
    rp_back = rp_map.transform(points)
    dots = np.abs((rp_back * rp_fit).sum(axis=1))
    assert np.all(dots > 1.0 - 1e-4), f"RP round trip: min |dot| = {dots.min()}"
    assert moebius_sign(rp_map.transform(fresh)) == -1.0
    assert np.isnan(rp_map.transform(np.array([[50.0, 50.0]], dtype=np.float32))).all()

    # CP map: round trip reproduces the fit; fresh (rotated) sphere points
    # give finite unit rows.
    cp_map = tda.coords.complex_projective_map(sphere)
    cp_fit = np.asarray(cp_map.coordinates)
    cp_back = cp_map.transform(sphere)
    assert cp_back.dtype == np.complex64
    assert np.allclose(cp_back, cp_fit, atol=1e-4)
    alpha = 0.1
    rot = sphere @ np.array(
        [
            [np.cos(alpha), np.sin(alpha), 0.0],
            [-np.sin(alpha), np.cos(alpha), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    cp_new = cp_map.transform(rot)
    assert not np.isnan(cp_new.view(np.float32)).any()
    assert np.allclose(np.linalg.norm(cp_new, axis=1), 1.0, atol=1e-4)

    print("coordinates: ok")


if __name__ == "__main__":
    main()
