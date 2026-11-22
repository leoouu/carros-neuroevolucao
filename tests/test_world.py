"""Colisão, checkpoints, eliminação e determinismo do mundo."""

from __future__ import annotations

import dataclasses

import numpy as np

from carros.baselines import heuristic_policy
from carros.sim.config import SimConfig
from carros.sim.sensors import ray_offsets
from carros.sim.track import Track
from carros.sim.world import Status, World


def _drive(world: World, steer: float, throttle: float, steps: int) -> None:
    n = world.n_cars
    for _ in range(steps):
        world.step(np.full(n, steer), np.full(n, throttle))


def test_driving_straight_in_corridor_survives(cfg: SimConfig, corridor: Track) -> None:
    world = World(corridor, cfg, eliminate_idle=False)
    _drive(world, 0.0, 1.0, 100)
    assert world.status[0] == Status.ALIVE


def test_steering_into_wall_crashes(cfg: SimConfig, corridor: Track) -> None:
    world = World(corridor, cfg, eliminate_idle=False)
    _drive(world, 1.0, 1.0, 200)
    assert world.status[0] == Status.CRASHED


def test_crashed_car_stops_moving(cfg: SimConfig, corridor: Track) -> None:
    world = World(corridor, cfg, eliminate_idle=False)
    _drive(world, 1.0, 1.0, 200)
    pos = world.state.positions.copy()
    _drive(world, 0.0, 1.0, 10)
    np.testing.assert_array_equal(world.state.positions, pos)


def test_fast_car_cannot_tunnel_through_wall(cfg: SimConfig, corridor: Track) -> None:
    """Com dt enorme o carro "pularia" a parede; o segmento varrido pega a colisão."""
    big = dataclasses.replace(cfg, dt=1.0, car=dataclasses.replace(cfg.car, turn_speed_ref=1.0))
    world = World(corridor, big, eliminate_idle=False)
    world.state = dataclasses.replace(
        world.state, speed=np.full(1, 300.0), heading=np.full(1, np.pi / 2)
    )
    world.step(np.zeros(1), np.ones(1))
    assert world.status[0] == Status.CRASHED


def test_checkpoints_are_counted_in_order(cfg: SimConfig, corridor: Track) -> None:
    world = World(corridor, cfg, eliminate_idle=False)
    # Velocidade final 300 px/s; em 3 s percorre bem mais que 300 px -> >= 3 checkpoints.
    _drive(world, 0.0, 1.0, 150)
    expected = int(world.state.x[0] // 100)
    assert world.checkpoints_passed[0] == expected
    assert world.next_checkpoint[0] == expected + 1


def test_stalled_car_is_eliminated(cfg: SimConfig, corridor: Track) -> None:
    world = World(corridor, cfg)
    _drive(world, 0.0, 0.0, int(cfg.elimination.stall_time / cfg.dt) + 2)
    assert world.status[0] == Status.STALLED


def test_car_without_progress_times_out(cfg: SimConfig, simple_track: Track) -> None:
    """Carro girando em círculo devagar: não bate, não para, mas não avança."""
    wide = dataclasses.replace(
        cfg, car=dataclasses.replace(cfg.car, max_speed=10.0, turn_speed_ref=1.0, turn_rate=2.0)
    )
    world = World(simple_track, wide)
    _drive(world, 1.0, 1.0, int(cfg.elimination.checkpoint_timeout / cfg.dt) + 2)
    assert world.status[0] == Status.TIMEOUT


def test_idle_rules_can_be_disabled(cfg: SimConfig, corridor: Track) -> None:
    world = World(corridor, cfg, eliminate_idle=False)
    _drive(world, 0.0, 0.0, 500)
    assert world.status[0] == Status.ALIVE


def test_heuristic_completes_a_lap_on_simple_track(cfg: SimConfig, simple_track: Track) -> None:
    """Sanidade da pista + física: uma heurística simples deve dar a volta."""
    world = World(simple_track, cfg)
    offsets = ray_offsets(cfg.sensors)
    while world.any_alive() and world.laps[0] == 0 and world.time < 60:
        world.step(*heuristic_policy(world.sensors(), offsets))
    assert world.laps[0] == 1
    assert world.checkpoints_passed[0] == simple_track.n_checkpoints
    assert not np.isnan(world.first_lap_time[0])


def test_simulation_is_deterministic(cfg: SimConfig, simple_track: Track) -> None:
    def run() -> np.ndarray:
        world = World(simple_track, cfg, n_cars=20)
        rng = np.random.default_rng(42)
        for _ in range(200):
            world.step(rng.uniform(-1, 1, 20), rng.uniform(-1, 1, 20))
        return np.column_stack([world.state.positions, world.status, world.checkpoints_passed])

    np.testing.assert_array_equal(run(), run())


def test_observe_shape(cfg: SimConfig, simple_track: Track) -> None:
    world = World(simple_track, cfg, n_cars=3)
    obs = world.observe()
    assert obs.shape == (3, cfg.sensors.count + 1)
