"""Fixtures compartilhadas: configuração conhecida e pistas de teste."""

from __future__ import annotations

import numpy as np
import pytest

from carros.sim.config import SimConfig, sim_config_from_dict
from carros.sim.track import Track, load_track

SIM_DICT = {
    "dt": 0.02,
    "car": {
        "length": 20.0,
        "width": 10.0,
        "max_speed": 300.0,
        "acceleration": 200.0,
        "brake": 400.0,
        "drag": 0.0,
        "turn_rate": 3.0,
        "turn_speed_ref": 50.0,
    },
    "sensors": {"count": 5, "fov_deg": 180.0, "max_range": 200.0},
    "elimination": {"stall_speed": 5.0, "stall_time": 1.0, "checkpoint_timeout": 3.0},
}


@pytest.fixture
def cfg() -> SimConfig:
    """Configuração com números redondos para contas de cabeça nos testes."""
    return sim_config_from_dict(SIM_DICT)


def make_corridor(length: float = 2000.0, width: float = 100.0, cp_every: float = 100.0) -> Track:
    """Corredor reto horizontal: paredes em y=0 e y=width, carro em (0, width/2) para +x.

    Checkpoints verticais a cada ``cp_every`` px; o checkpoint 0 fica em x=0 (largada).
    """
    half = length / 2
    walls = np.array([[-half, 0.0, half, 0.0], [-half, width, half, width]])
    xs = np.arange(0.0, half, cp_every)
    checkpoints = np.column_stack([xs, np.zeros_like(xs), xs, np.full_like(xs, width)])
    outer = np.array([[-half, 0.0], [half, 0.0], [half, width], [-half, width]])
    center = np.column_stack([xs, np.full_like(xs, width / 2)])
    return Track(
        name="corridor",
        width=width,
        centerline=center,
        inner=outer,
        outer=outer,
        walls=walls,
        checkpoints=checkpoints,
        start_pos=np.array([0.0, width / 2]),
        start_heading=0.0,
    )


@pytest.fixture
def corridor() -> Track:
    """Corredor reto padrão."""
    return make_corridor()


@pytest.fixture(scope="session")
def simple_track() -> Track:
    """A pista simples real do projeto."""
    return load_track("configs/tracks/simple.yaml")
