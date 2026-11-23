"""Gerador de pistas aleatórias (determinístico dada a semente).

Ideia: espalhar ``n`` pontos em volta de um círculo, cada um com ângulo e raio
sorteados. Raios variados criam "barrigas" para dentro e curvas fechadas; a
suavização de Chaikin arredonda tudo. Se a pista sair inválida (borda dobrando,
partes se encostando), sorteia de novo — a validação de ``build_track`` é o
filtro.

O sentido de corrida também é sorteado: se toda pista de treino virasse sempre
para o mesmo lado, a rede poderia aprender só a virar para a direita.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from carros.sim.track import Track, track_from_dict


@dataclass(frozen=True)
class GeneratorParams:
    """Faixas de sorteio para as pistas (lidas de YAML)."""

    n_points: list[int]  # [mín, máx]
    radius: list[float]  # [mín, máx] em px
    angle_jitter: float  # fração do espaçamento angular entre pontos
    width: list[float]  # [mín, máx]
    smoothing: int
    spacing: float
    checkpoint_every: int
    max_tries: int


def load_generator_params(path: str | Path) -> GeneratorParams:
    """Lê os parâmetros do gerador de um YAML."""
    with open(path, encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh)
    expected = {f.name for f in fields(GeneratorParams)}
    if set(data) != expected:
        raise ValueError(f"gerador: esperado {sorted(expected)}, recebido {sorted(data)}")
    return GeneratorParams(**data)


def _random_centerline(params: GeneratorParams, rng: np.random.Generator) -> np.ndarray:
    n = int(rng.integers(params.n_points[0], params.n_points[1] + 1))
    step = 2 * np.pi / n
    angles = np.arange(n) * step + rng.uniform(-1, 1, n) * params.angle_jitter * step
    radii = rng.uniform(params.radius[0], params.radius[1], n)
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    pts = np.roll(pts, -int(rng.integers(n)), axis=0)  # largada num ponto qualquer
    if rng.random() < 0.5:
        pts = pts[::-1]  # inverte o sentido de corrida
    return pts - pts.min(axis=0) + 60.0  # coordenadas positivas, com margem


def generate_track_dict(name: str, params: GeneratorParams, seed: int) -> dict[str, Any]:
    """Sorteia uma pista válida e devolve o dicionário no formato do YAML de pista.

    Raises:
        RuntimeError: se nenhuma tentativa gerar uma pista válida.
    """
    rng = np.random.default_rng(seed)
    for _ in range(params.max_tries):
        data = {
            "name": name,
            "width": round(float(rng.uniform(params.width[0], params.width[1])), 1),
            "smoothing": params.smoothing,
            "spacing": params.spacing,
            "checkpoint_every": params.checkpoint_every,
            "control_points": np.round(_random_centerline(params, rng), 1).tolist(),
        }
        try:
            track_from_dict(data)
        except ValueError:
            continue
        return data
    raise RuntimeError(f"Nenhuma pista válida em {params.max_tries} tentativas (seed {seed})")


def generate_track(name: str, params: GeneratorParams, seed: int) -> Track:
    """Atalho: sorteia e já constrói a ``Track``."""
    return track_from_dict(generate_track_dict(name, params, seed))
