"""Física do carro e sensores."""

from __future__ import annotations

import numpy as np
import pytest

from carros.sim.car import CarState, car_corners, step_cars
from carros.sim.config import SimConfig
from carros.sim.sensors import ray_offsets, read_sensors
from carros.sim.track import Track


def _still(n: int = 1) -> CarState:
    return CarState.spawn(n, np.array([0.0, 0.0]), 0.0)


def test_throttle_accelerates_and_respects_max_speed(cfg: SimConfig) -> None:
    s = _still()
    s = step_cars(s, np.zeros(1), np.ones(1), cfg.car, cfg.dt)
    assert s.speed[0] == pytest.approx(cfg.car.acceleration * cfg.dt)
    for _ in range(500):
        s = step_cars(s, np.zeros(1), np.ones(1), cfg.car, cfg.dt)
    assert s.speed[0] == pytest.approx(cfg.car.max_speed)
    assert s.y[0] == pytest.approx(0.0)  # sem volante, anda reto


def test_brake_never_goes_negative(cfg: SimConfig) -> None:
    s = step_cars(_still(), np.zeros(1), -np.ones(1), cfg.car, cfg.dt)
    assert s.speed[0] == 0.0


def test_stopped_car_cannot_turn(cfg: SimConfig) -> None:
    s = step_cars(_still(), np.ones(1), np.zeros(1), cfg.car, cfg.dt)
    assert s.heading[0] == 0.0


def test_positive_steer_increases_heading(cfg: SimConfig) -> None:
    s = CarState(np.zeros(1), np.zeros(1), np.zeros(1), np.full(1, 100.0))
    s = step_cars(s, np.ones(1), np.zeros(1), cfg.car, cfg.dt)
    assert s.heading[0] > 0


def test_inactive_cars_are_frozen(cfg: SimConfig) -> None:
    s = CarState(np.zeros(2), np.zeros(2), np.zeros(2), np.full(2, 100.0))
    s2 = step_cars(s, np.zeros(2), np.ones(2), cfg.car, cfg.dt, active=np.array([True, False]))
    assert s2.x[0] > 0 and s2.x[1] == 0.0


def test_corners_match_dimensions(cfg: SimConfig) -> None:
    c = car_corners(_still(), cfg.car)[0]
    assert np.ptp(c[:, 0]) == pytest.approx(cfg.car.length)
    assert np.ptp(c[:, 1]) == pytest.approx(cfg.car.width)


def test_ray_offsets_are_symmetric(cfg: SimConfig) -> None:
    off = ray_offsets(cfg.sensors)
    assert len(off) == cfg.sensors.count
    np.testing.assert_allclose(off, -off[::-1])
    assert off[0] == pytest.approx(-np.pi / 2)


def test_sensors_in_corridor(cfg: SimConfig, corridor: Track) -> None:
    """No centro de um corredor de 100 px: laterais medem 50 px, frente está livre."""
    state = CarState.spawn(1, corridor.start_pos, 0.0)
    r = read_sensors(state, corridor.walls, cfg.sensors)[0] * cfg.sensors.max_range
    assert r[0] == pytest.approx(50.0)  # -90°
    assert r[-1] == pytest.approx(50.0)  # +90°
    assert r[2] == pytest.approx(cfg.sensors.max_range)  # frente
    assert r[1] == pytest.approx(50.0 / np.sin(np.pi / 4))  # -45°


def test_sensors_are_normalized(cfg: SimConfig, simple_track: Track) -> None:
    state = CarState.spawn(5, simple_track.start_pos, simple_track.start_heading)
    r = read_sensors(state, simple_track.walls, cfg.sensors)
    assert r.shape == (5, cfg.sensors.count)
    assert np.all((r >= 0) & (r <= 1))
