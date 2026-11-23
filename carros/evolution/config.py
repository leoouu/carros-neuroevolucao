"""Configuração de um experimento de treino (um YAML por experimento)."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


@dataclass(frozen=True)
class EpisodeParams:
    """Duração de cada avaliação."""

    max_time: float
    target_laps: int


@dataclass(frozen=True)
class NetworkParams:
    """Arquitetura da rede: neurônios por camada oculta e escala dos pesos iniciais."""

    hidden: list[int]
    init_scale: float


@dataclass(frozen=True)
class GAParams:
    """Hiperparâmetros do algoritmo genético."""

    population: int
    elite: int
    tournament_size: int
    crossover: str  # "uniform" | "single_point"
    crossover_rate: float
    mutation_rate: float
    mutation_sigma: float
    sigma_decay: float
    sigma_min: float
    diversity_threshold: float
    diversity_boost: float
    inject_fraction: float


@dataclass(frozen=True)
class FitnessParams:
    """Pesos do fitness."""

    time_penalty: float
    lap_bonus: float


@dataclass(frozen=True)
class TrainConfig:
    """Experimento completo."""

    name: str
    seed: int
    generations: int
    tracks: list[str]  # pistas de treino (arquivos .yaml ou pastas)
    validation_tracks: list[str]  # pistas NUNCA vistas no treino (podem ser [])
    validate_every: int  # a cada quantas gerações medir generalização
    workers: int  # processos em paralelo (1 = sem paralelismo)
    sim: str
    output_dir: str
    episode: EpisodeParams
    network: NetworkParams
    ga: GAParams
    fitness: FitnessParams


def _build(cls: type[T], data: dict[str, Any], section: str) -> T:
    expected = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    missing, extra = expected - data.keys(), data.keys() - expected
    if missing or extra:
        raise ValueError(f"[{section}] faltando={sorted(missing)} sobrando={sorted(extra)}")
    return cls(**data)


def train_config_from_dict(data: dict[str, Any]) -> TrainConfig:
    """Valida e constrói a configuração a partir do dicionário do YAML."""
    nested = {
        "episode": EpisodeParams,
        "network": NetworkParams,
        "ga": GAParams,
        "fitness": FitnessParams,
    }
    built: dict[str, Any] = {k: _build(cls, data[k], k) for k, cls in nested.items()}
    flat = {k: v for k, v in data.items() if k not in nested}
    flat["tracks"] = expand_track_paths(flat.get("tracks", []))
    flat["validation_tracks"] = expand_track_paths(flat.get("validation_tracks") or [])
    cfg = _build(TrainConfig, {**flat, **built}, "raiz")
    if not cfg.tracks:
        raise ValueError("nenhuma pista de treino")
    overlap = set(cfg.tracks) & set(cfg.validation_tracks)
    if overlap:
        raise ValueError(f"pistas de validação também estão no treino: {sorted(overlap)}")
    if cfg.ga.crossover not in ("uniform", "single_point"):
        raise ValueError(f"crossover desconhecido: {cfg.ga.crossover}")
    if not 0 <= cfg.ga.elite < cfg.ga.population:
        raise ValueError("elite deve estar entre 0 e population - 1")
    return cfg


def expand_track_paths(entries: list[str]) -> list[str]:
    """Troca cada pasta pelos ``.yaml`` dentro dela (ordenados); arquivos ficam como estão."""
    out: list[str] = []
    for entry in entries:
        path = Path(entry)
        if path.is_dir():
            out.extend(p.as_posix() for p in sorted(path.glob("*.yaml")))
        else:
            out.append(path.as_posix())
    return out


def load_train_config(path: str | Path) -> tuple[TrainConfig, dict[str, Any]]:
    """Lê o YAML; devolve a config tipada e o dicionário cru (para salvar junto da execução)."""
    with open(path, encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)
    return train_config_from_dict(raw), raw
