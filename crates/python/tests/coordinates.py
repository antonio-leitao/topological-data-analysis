"""Coordinate API integration checks."""

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


def winding(angles):
    """Winding number of an angle sequence around a sampled closed loop."""
    diffs = np.diff(np.append(angles, angles[0]))
    wrapped = (diffs + np.pi) % (2.0 * np.pi) - np.pi
    return float(wrapped.sum() / (2.0 * np.pi))


def check_single_class(points):
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

    w = winding(angles[:, 0])
    assert abs(abs(w) - 1.0) < 1e-3, f"expected winding 1, got {w}"

    again = tda.coords.circular(points)
    assert np.array_equal(angles, np.asarray(again.coordinates))


def check_distance_matrix(points):
    diff = points[:, None, :] - points[None, :, :]
    dist = np.sqrt((diff * diff).sum(-1)).astype(np.float32)
    res = tda.coords.circular(dist, distance_matrix=True)
    w = winding(np.asarray(res.coordinates)[:, 0])
    assert abs(abs(w) - 1.0) < 1e-3, f"distance-matrix winding: {w}"


def check_toroidal():
    """Two side-by-side circles: each column must wind exactly one of them."""
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
        [[winding(angles[:per, c]), winding(angles[per:, c])] for c in range(2)]
    )
    winds_once = np.abs(np.abs(w) - 1.0) < 1e-3
    is_flat = np.abs(w) < 1e-3
    diag = winds_once[0, 0] and is_flat[0, 1] and is_flat[1, 0] and winds_once[1, 1]
    anti = is_flat[0, 0] and winds_once[0, 1] and winds_once[1, 0] and is_flat[1, 1]
    assert diag or anti, f"each column must wind exactly one circle, got {w}"


def check_quotient(points):
    res = tda.coords.circular(points, quotient=True)
    angles = np.asarray(res.coordinates)
    assert angles.shape == (len(points), 1)
    w = winding(angles[:, 0])
    assert abs(abs(w) - 1.0) < 1e-3, f"quotient winding: {w}"


def check_rejections(points):
    n = len(points)

    empty = tda.coords.circular(points, classes=[])
    assert np.asarray(empty.coordinates).shape == (n, 0)
    assert empty.groups == []

    try:
        tda.coords.circular(points, prime=4)
        raise AssertionError("expected ValueError for prime=4")
    except ValueError:
        pass

    try:
        tda.coords.circular(points, classes=[99])
        raise AssertionError("expected ValueError for class rank 99")
    except ValueError:
        pass

    class FakeSparse:
        def tocoo(self):  # pragma: no cover
            return self

    try:
        tda.coords.circular(FakeSparse())
        raise AssertionError("expected TypeError for sparse input")
    except TypeError:
        pass


def check_fitted_map(points):
    n = len(points)
    model = tda.coords.circular_map(points)
    assert isinstance(model, tda.coords.CircularMap)

    train_values = np.asarray(model.coordinates)
    assert train_values.shape == (n, 1)
    plain = tda.coords.circular(points)
    assert np.array_equal(train_values, np.asarray(plain.coordinates))

    back = model.transform(points)
    assert back.shape == (n, 1) and not np.isnan(back).any()
    assert abs(abs(winding(back[:, 0])) - 1.0) < 1e-3

    phases = 2.0 * np.pi * (np.arange(60) + 0.37) / 60
    fresh = np.stack([np.cos(phases), np.sin(phases)], axis=1).astype(np.float32)
    values = model.transform(fresh)
    assert not np.isnan(values).any()
    assert abs(abs(winding(values[:, 0])) - 1.0) < 1e-3

    far = model.transform(np.array([[100.0, 100.0]], dtype=np.float32))
    assert np.isnan(far).all()

    try:
        model.transform(np.zeros((3, 5), dtype=np.float32))
        raise AssertionError("expected ValueError for wrong ambient dimension")
    except ValueError:
        pass


def main():
    points = circle(48)

    check_single_class(points)
    check_distance_matrix(points)
    check_toroidal()
    check_quotient(points)
    check_rejections(points)
    check_fitted_map(points)

    print("coordinates: ok")


if __name__ == "__main__":
    main()
