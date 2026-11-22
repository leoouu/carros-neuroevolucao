"""Modelo cinemático do carro, vetorizado para ``N`` carros.

Por que cinemático (e não dinâmico, com pneus e derrapagem)? Porque o objetivo é
a evolução, não a física: um modelo simples é determinístico, barato e fácil de
entender, e ainda assim exige que a rede aprenda a frear antes das curvas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from carros.sim.config import CarParams
from carros.sim.geometry import BoolArray, FloatArray


@dataclass(frozen=True)
class CarState:
    """Estado de ``N`` carros em arrays ``(N,)``. Imutável: ``step_cars`` devolve um novo."""

    x: FloatArray
    y: FloatArray
    heading: FloatArray
    speed: FloatArray

    @property
    def positions(self) -> FloatArray:
        """``(N, 2)`` posições."""
        return np.column_stack([self.x, self.y])

    def subset(self, idx: np.ndarray) -> CarState:
        """Estado apenas dos carros em ``idx``."""
        return CarState(self.x[idx], self.y[idx], self.heading[idx], self.speed[idx])

    @classmethod
    def spawn(cls, n: int, pos: FloatArray, heading: float) -> CarState:
        """``n`` carros parados na mesma posição e direção."""
        return cls(
            x=np.full(n, float(pos[0])),
            y=np.full(n, float(pos[1])),
            heading=np.full(n, heading),
            speed=np.zeros(n),
        )


def step_cars(
    state: CarState,
    steer: FloatArray,
    throttle: FloatArray,
    params: CarParams,
    dt: float,
    active: BoolArray | None = None,
) -> CarState:
    """Avança a física um passo.

    Args:
        steer: ``(N,)`` em ``[-1, 1]``; positivo aumenta o ângulo.
        throttle: ``(N,)`` em ``[-1, 1]``; positivo acelera, negativo freia.
        active: máscara ``(N,)``; carros inativos (eliminados) ficam congelados.
    """
    steer = np.clip(steer, -1.0, 1.0)
    throttle = np.clip(throttle, -1.0, 1.0)

    accel = np.where(throttle >= 0, throttle * params.acceleration, throttle * params.brake)
    speed = state.speed + (accel - params.drag * state.speed) * dt
    speed = np.clip(speed, 0.0, params.max_speed)

    # Giro proporcional à velocidade em baixa: um carro parado não gira no lugar.
    grip = np.minimum(1.0, speed / params.turn_speed_ref)
    heading = state.heading + steer * params.turn_rate * grip * dt
    x = state.x + np.cos(heading) * speed * dt
    y = state.y + np.sin(heading) * speed * dt

    if active is not None:
        x = np.where(active, x, state.x)
        y = np.where(active, y, state.y)
        heading = np.where(active, heading, state.heading)
        speed = np.where(active, speed, 0.0)
    return CarState(x=x, y=y, heading=heading, speed=speed)


def car_corners(state: CarState, params: CarParams) -> FloatArray:
    """``(N, 4, 2)`` cantos do carro (frente-esq, frente-dir, trás-dir, trás-esq)."""
    hl, hw = params.length / 2.0, params.width / 2.0
    local = np.array([[hl, -hw], [hl, hw], [-hl, hw], [-hl, -hw]])  # (4, 2)
    c, s = np.cos(state.heading)[:, None], np.sin(state.heading)[:, None]
    wx = state.x[:, None] + local[:, 0] * c - local[:, 1] * s
    wy = state.y[:, None] + local[:, 0] * s + local[:, 1] * c
    return np.stack([wx, wy], axis=-1)


def car_edges(corners: FloatArray) -> FloatArray:
    """``(N, 4, 4)`` os 4 lados de cada carro como segmentos."""
    return np.concatenate([corners, np.roll(corners, -1, axis=1)], axis=-1)
