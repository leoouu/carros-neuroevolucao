"""Mundo: ``N`` carros numa pista, com colisão, checkpoints e eliminação."""

from __future__ import annotations

from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

from carros.sim.car import CarState, car_corners, car_edges, step_cars
from carros.sim.config import SimConfig
from carros.sim.geometry import (
    BoolArray,
    FloatArray,
    segments_intersect_pairwise,
)
from carros.sim.sensors import read_sensors
from carros.sim.spatial import build_wall_grid
from carros.sim.track import Track

IntArray = NDArray[np.int64]


class Status(IntEnum):
    """Situação de cada carro."""

    ALIVE = 0
    CRASHED = 1
    STALLED = 2
    TIMEOUT = 3


class World:
    """Simulação headless e determinística de ``n_cars`` carros.

    Uso típico::

        world = World(track, cfg, n_cars=100)
        while world.any_alive():
            obs = world.observe()          # (N, R+1)
            world.step(steer, throttle)    # arrays (N,)

    Args:
        eliminate_idle: se False, desliga as regras de "parado" e "sem checkpoint"
            (útil no modo manual); a colisão continua valendo.
    """

    def __init__(
        self, track: Track, cfg: SimConfig, n_cars: int = 1, eliminate_idle: bool = True
    ) -> None:
        self.track = track
        self.cfg = cfg
        self.n_cars = n_cars
        self.eliminate_idle = eliminate_idle
        car = cfg.car
        # Folga da colisão: meia diagonal do carro + o máximo que ele anda num passo.
        near_margin = float(np.hypot(car.length, car.width)) / 2 + car.max_speed * cfg.dt + 1.0
        self.grid = build_wall_grid(
            track.walls, cell_size=40.0, near_margin=near_margin, ray_range=cfg.sensors.max_range
        )
        self.reset()

    def reset(self) -> None:
        """Recoloca todos os carros na largada."""
        n = self.n_cars
        self.state = CarState.spawn(n, self.track.start_pos, self.track.start_heading)
        self.status: IntArray = np.full(n, Status.ALIVE, dtype=np.int64)
        # O carro nasce sobre o checkpoint 0 (largada), então o próximo é o 1.
        self.next_checkpoint: IntArray = np.full(n, 1 % self.track.n_checkpoints, dtype=np.int64)
        self.checkpoints_passed: IntArray = np.zeros(n, dtype=np.int64)
        self.laps: IntArray = np.zeros(n, dtype=np.int64)
        self.first_lap_time: FloatArray = np.full(n, np.nan)
        self.time_since_checkpoint: FloatArray = np.zeros(n)
        self.stall_time: FloatArray = np.zeros(n)
        self.time = 0.0
        self.steps = 0

    # ----------------------------------------------------------------- consultas
    @property
    def alive(self) -> BoolArray:
        """Máscara ``(N,)`` dos carros ainda em jogo."""
        result: BoolArray = self.status == Status.ALIVE
        return result

    def any_alive(self) -> bool:
        """Ainda existe carro rodando?"""
        return bool(np.any(self.alive))

    def sensors(self) -> FloatArray:
        """``(N, R)`` leituras normalizadas dos raios (carros eliminados leem 0)."""
        out = np.zeros((self.n_cars, self.cfg.sensors.count))
        idx = np.flatnonzero(self.alive)
        if idx.size:
            sub = self.state.subset(idx)
            out[idx] = read_sensors(sub, self.grid.far_walls(sub.positions), self.cfg.sensors)
        return out

    def observe(self) -> FloatArray:
        """``(N, R+1)`` entradas da rede: sensores + velocidade normalizada."""
        speed = self.state.speed / self.cfg.car.max_speed
        return np.column_stack([self.sensors(), speed])

    # ----------------------------------------------------------------- dinâmica
    def step(self, steer: FloatArray, throttle: FloatArray) -> None:
        """Avança um passo de ``dt`` para todos os carros vivos."""
        alive = self.alive
        prev = self.state
        self.state = step_cars(
            prev,
            np.asarray(steer, dtype=np.float64),
            np.asarray(throttle, dtype=np.float64),
            self.cfg.car,
            self.cfg.dt,
            active=alive,
        )
        self.time += self.cfg.dt
        self.steps += 1

        # Segmento percorrido neste passo: usado no checkpoint e contra "tunneling"
        # (em alta velocidade o carro poderia pular por cima de uma parede fina).
        sweep = np.column_stack([prev.x, prev.y, self.state.x, self.state.y])
        self._update_checkpoints(sweep, alive)
        self._update_collisions(sweep, alive)
        if self.eliminate_idle:
            self._update_idle(alive)

    def _update_checkpoints(self, sweep: FloatArray, alive: BoolArray) -> None:
        cps = self.track.checkpoints[self.next_checkpoint]
        crossed = segments_intersect_pairwise(sweep, cps) & alive
        dt = self.cfg.dt
        self.time_since_checkpoint = np.where(crossed, 0.0, self.time_since_checkpoint + dt)
        lap_done = crossed & (self.next_checkpoint == 0)
        self.laps += lap_done
        first = lap_done & np.isnan(self.first_lap_time)
        self.first_lap_time = np.where(first, self.time, self.first_lap_time)
        self.checkpoints_passed += crossed
        advanced = (self.next_checkpoint + 1) % self.track.n_checkpoints
        self.next_checkpoint = np.where(crossed, advanced, self.next_checkpoint)

    def _update_collisions(self, sweep: FloatArray, alive: BoolArray) -> None:
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            return
        sub = self.state.subset(idx)
        walls = self.grid.near_walls(sub.positions)  # (M, K, 4) só paredes próximas
        edges = car_edges(car_corners(sub, self.cfg.car))  # (M, 4, 4)
        body_hit = segments_intersect_pairwise(edges[:, :, None, :], walls[:, None]).any(
            axis=(1, 2)
        )
        sweep_hit = segments_intersect_pairwise(sweep[idx][:, None, :], walls).any(axis=1)
        self.status[idx[body_hit | sweep_hit]] = Status.CRASHED

    def _update_idle(self, alive: BoolArray) -> None:
        elim = self.cfg.elimination
        still = alive & (self.status == Status.ALIVE)
        slow = self.state.speed < elim.stall_speed
        self.stall_time = np.where(slow & still, self.stall_time + self.cfg.dt, 0.0)
        self.status[still & (self.stall_time >= elim.stall_time)] = Status.STALLED
        still = self.status == Status.ALIVE
        self.status[still & (self.time_since_checkpoint >= elim.checkpoint_timeout)] = (
            Status.TIMEOUT
        )
