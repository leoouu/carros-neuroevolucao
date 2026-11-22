"""Configuração da simulação carregada de YAML (nada de hiperparâmetro hardcoded)."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


@dataclass(frozen=True)
class CarParams:
    """Física do carro (unidades: pixels e segundos)."""

    length: float
    width: float
    max_speed: float
    acceleration: float
    brake: float
    drag: float
    turn_rate: float
    turn_speed_ref: float


@dataclass(frozen=True)
class SensorParams:
    """Raios de distância: quantidade, abertura total e alcance."""

    count: int
    fov_deg: float
    max_range: float


@dataclass(frozen=True)
class EliminationParams:
    """Regras que eliminam um carro além da colisão."""

    stall_speed: float
    stall_time: float
    checkpoint_timeout: float


@dataclass(frozen=True)
class SimConfig:
    """Configuração completa de uma simulação."""

    dt: float
    car: CarParams
    sensors: SensorParams
    elimination: EliminationParams


def _build(cls: type[T], data: dict[str, Any], section: str) -> T:
    """Instancia um dataclass exigindo exatamente as chaves esperadas."""
    expected = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    missing, extra = expected - data.keys(), data.keys() - expected
    if missing or extra:
        raise ValueError(f"[{section}] faltando={sorted(missing)} sobrando={sorted(extra)}")
    return cls(**data)


def sim_config_from_dict(data: dict[str, Any]) -> SimConfig:
    """Constrói ``SimConfig`` a partir de um dicionário (ex.: YAML já lido)."""
    return SimConfig(
        dt=float(data["dt"]),
        car=_build(CarParams, data["car"], "car"),
        sensors=_build(SensorParams, data["sensors"], "sensors"),
        elimination=_build(EliminationParams, data["elimination"], "elimination"),
    )


def load_sim_config(path: str | Path) -> SimConfig:
    """Lê um YAML de simulação."""
    with open(path, encoding="utf-8") as fh:
        return sim_config_from_dict(yaml.safe_load(fh))
