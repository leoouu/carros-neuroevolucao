"""Raycast e interseção de segmentos."""

from __future__ import annotations

import numpy as np
import pytest

from carros.sim.geometry import (
    chaikin_smooth,
    ray_segment_distances,
    segments_intersect,
    segments_intersect_pairwise,
    signed_area,
)

WALL_X10 = np.array([[10.0, -5.0, 10.0, 5.0]])  # parede vertical em x=10


def _ray(origin: tuple[float, float], angle: float, walls: np.ndarray, rng: float = 100.0) -> float:
    o = np.array([origin])
    d = np.array([[[np.cos(angle), np.sin(angle)]]])
    return float(ray_segment_distances(o, d, walls, rng)[0, 0])


def test_ray_hits_wall_at_exact_distance() -> None:
    assert _ray((0, 0), 0.0, WALL_X10) == pytest.approx(10.0)


def test_ray_hits_wall_diagonally() -> None:
    # A 45°, chega em x=10 com y=10 -> fora da parede (y em [-5, 5]); a 26.57° acerta.
    assert _ray((0, 0), np.pi / 4, WALL_X10) == 100.0
    angle = np.arctan2(5, 10)
    assert _ray((0, 0), angle, WALL_X10) == pytest.approx(np.hypot(10, 5))


def test_ray_pointing_away_returns_max_range() -> None:
    assert _ray((0, 0), np.pi, WALL_X10) == 100.0


def test_parallel_ray_misses() -> None:
    horizontal = np.array([[0.0, 5.0, 50.0, 5.0]])
    assert _ray((0, 0), 0.0, horizontal) == 100.0


def test_ray_beyond_range_is_clipped() -> None:
    assert _ray((0, 0), 0.0, WALL_X10, rng=4.0) == 4.0


def test_ray_returns_nearest_of_several_walls() -> None:
    walls = np.array([[30.0, -5, 30, 5], [10.0, -5, 10, 5], [20.0, -5, 20, 5]])
    assert _ray((0, 0), 0.0, walls) == pytest.approx(10.0)


def test_ray_batch_shapes_and_per_car_walls_match_shared_walls() -> None:
    rng = np.random.default_rng(1)
    walls = rng.uniform(-50, 50, (30, 4))
    origins = rng.uniform(-20, 20, (6, 2))
    ang = rng.uniform(-np.pi, np.pi, (6, 4))
    dirs = np.stack([np.cos(ang), np.sin(ang)], axis=-1)
    shared = ray_segment_distances(origins, dirs, walls, 80.0)
    per_car = ray_segment_distances(origins, dirs, np.broadcast_to(walls, (6, 30, 4)), 80.0)
    assert shared.shape == (6, 4)
    np.testing.assert_allclose(shared, per_car)


def test_segments_cross() -> None:
    a = np.array([[0.0, 0, 10, 10]])
    b = np.array([[0.0, 10, 10, 0], [20.0, 0, 30, 0], [0.0, 1, 10, 11]])
    assert segments_intersect(a, b).tolist() == [[True, False, False]]


def test_segments_touching_at_endpoint_count_as_intersecting() -> None:
    a = np.array([[0.0, 0, 10, 0]])
    b = np.array([[10.0, 0, 10, 10]])
    assert segments_intersect(a, b)[0, 0]


def test_pairwise_matches_diagonal_of_matrix() -> None:
    rng = np.random.default_rng(2)
    a, b = rng.uniform(0, 10, (20, 4)), rng.uniform(0, 10, (20, 4))
    np.testing.assert_array_equal(
        segments_intersect_pairwise(a, b), np.diag(segments_intersect(a, b))
    )


def test_zero_length_segment_is_never_hit() -> None:
    null = np.array([[5.0, 0, 5, 0]])
    assert _ray((0, 0), 0.0, null) == 100.0
    assert not segments_intersect(np.array([[0.0, 0, 10, 0]]), null)[0, 0]


def test_chaikin_keeps_orientation_and_shrinks_square() -> None:
    square = np.array([[0.0, 0], [10, 0], [10, 10], [0, 10]])
    smooth = chaikin_smooth(square, 2)
    assert len(smooth) == 16
    assert signed_area(smooth) > 0
    assert 0 < signed_area(smooth) < signed_area(square)
