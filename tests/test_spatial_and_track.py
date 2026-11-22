"""Grade espacial (deve ser idêntica à força bruta) e construção de pistas."""

from __future__ import annotations

import numpy as np
import pytest

from carros.sim.geometry import ray_segment_distances, segments_intersect
from carros.sim.spatial import build_wall_grid
from carros.sim.track import Track, build_track


def test_grid_raycast_equals_brute_force(simple_track: Track) -> None:
    grid = build_wall_grid(simple_track.walls, 40.0, near_margin=20.0, ray_range=250.0)
    rng = np.random.default_rng(0)
    xmin, ymin, xmax, ymax = simple_track.bounds()
    origins = rng.uniform([xmin, ymin], [xmax, ymax], (300, 2))
    ang = rng.uniform(-np.pi, np.pi, (300, 7))
    dirs = np.stack([np.cos(ang), np.sin(ang)], axis=-1)
    brute = ray_segment_distances(origins, dirs, simple_track.walls, 250.0)
    fast = ray_segment_distances(origins, dirs, grid.far_walls(origins), 250.0)
    np.testing.assert_allclose(fast, brute)


def test_grid_near_walls_contains_every_wall_within_margin(simple_track: Track) -> None:
    grid = build_wall_grid(simple_track.walls, 40.0, near_margin=20.0, ray_range=250.0)
    rng = np.random.default_rng(1)
    xmin, ymin, xmax, ymax = simple_track.bounds()
    centers = rng.uniform([xmin, ymin], [xmax, ymax], (300, 2))
    # Segmentos curtos (<20 px) em volta de cada centro: o que a força bruta acha, a grade acha.
    ang = rng.uniform(-np.pi, np.pi, 300)
    segs = np.column_stack([centers, centers + 19.0 * np.column_stack([np.cos(ang), np.sin(ang)])])
    brute = segments_intersect(segs, simple_track.walls).any(axis=1)
    near = grid.near_walls(centers)
    fast = np.array([segments_intersect(s[None], w).any() for s, w in zip(segs, near, strict=True)])
    np.testing.assert_array_equal(fast, brute)


def test_simple_track_is_consistent(simple_track: Track) -> None:
    assert simple_track.n_checkpoints > 10
    assert len(simple_track.walls) == len(simple_track.inner) + len(simple_track.outer)
    # Largada aponta para o checkpoint 1.
    cp1_mid = simple_track.checkpoints[1].reshape(2, 2).mean(axis=0)
    to_cp1 = cp1_mid - simple_track.start_pos
    heading = np.array([np.cos(simple_track.start_heading), np.sin(simple_track.start_heading)])
    assert np.dot(to_cp1, heading) > 0


def test_track_too_tight_for_its_width_is_rejected() -> None:
    tight = np.array([[0.0, 0], [100, 0], [100, 30], [0, 30]])
    with pytest.raises(ValueError, match="auto-intersecta|se cruzam"):
        build_track("apertada", tight, width=80.0, smoothing=3, spacing=5.0)
