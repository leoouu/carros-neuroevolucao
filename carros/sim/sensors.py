"""Sensores de distância por raycast."""

from __future__ import annotations

import numpy as np

from carros.sim.car import CarState
from carros.sim.config import SensorParams
from carros.sim.geometry import FloatArray, ray_segment_distances


def ray_offsets(params: SensorParams) -> FloatArray:
    """Ângulos (rad) dos raios relativos à frente do carro, distribuídos no leque."""
    if params.count == 1:
        return np.zeros(1)
    half = np.deg2rad(params.fov_deg) / 2.0
    return np.linspace(-half, half, params.count)


def ray_directions(state: CarState, params: SensorParams) -> FloatArray:
    """``(N, R, 2)`` vetores unitários de cada raio."""
    angles = state.heading[:, None] + ray_offsets(params)[None, :]
    return np.stack([np.cos(angles), np.sin(angles)], axis=-1)


def read_sensors(state: CarState, walls: FloatArray, params: SensorParams) -> FloatArray:
    """``(N, R)`` distância até a parede, normalizada: 0 = encostado, 1 = livre até o alcance.

    Normalizar em [0, 1] importa para a rede neural: entradas na mesma escala
    evitam que um sensor "grite" mais alto que outro só por causa da unidade.
    """
    dirs = ray_directions(state, params)
    dist = ray_segment_distances(state.positions, dirs, walls, params.max_range)
    return dist / params.max_range
