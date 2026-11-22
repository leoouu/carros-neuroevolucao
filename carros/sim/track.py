"""Pista 2D: bordas interna/externa, paredes e checkpoints.

Decisão: a pista é descrita no YAML por uma **linha central** (pontos de controle)
e uma **largura**. As bordas são obtidas deslocando a linha central para os dois
lados ao longo da normal. Vantagens: (1) checkpoints saem naturalmente como
segmentos transversais à pista; (2) um gerador aleatório (Fase 3) só precisa
sortear uma linha central; (3) as duas bordas nunca ficam "desalinhadas".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from carros.sim.geometry import (
    FloatArray,
    chaikin_smooth,
    polyline_to_segments,
    resample_closed,
    segments_intersect,
    signed_area,
)


@dataclass(frozen=True)
class Track:
    """Pista pronta para simulação.

    Attributes:
        name: identificador da pista.
        centerline: ``(P, 2)`` pontos da linha central, no sentido de corrida.
        inner: ``(P, 2)`` borda interna (polígono fechado).
        outer: ``(P, 2)`` borda externa (polígono fechado).
        walls: ``(2P, 4)`` todos os segmentos de parede (interna + externa).
        checkpoints: ``(C, 4)`` segmentos transversais numerados; o 0 é a largada.
        start_pos: ``(2,)`` posição inicial dos carros.
        start_heading: ângulo inicial (rad).
    """

    name: str
    width: float
    centerline: FloatArray
    inner: FloatArray
    outer: FloatArray
    walls: FloatArray
    checkpoints: FloatArray
    start_pos: FloatArray
    start_heading: float

    @property
    def n_checkpoints(self) -> int:
        """Número de checkpoints por volta."""
        return len(self.checkpoints)

    def bounds(self) -> tuple[float, float, float, float]:
        """Retângulo envolvente ``(xmin, ymin, xmax, ymax)`` da borda externa."""
        xmin, ymin = self.outer.min(axis=0)
        xmax, ymax = self.outer.max(axis=0)
        return float(xmin), float(ymin), float(xmax), float(ymax)


def _normals(points: FloatArray) -> FloatArray:
    """Normais unitárias de um polígono fechado (tangente por diferença central)."""
    tangent = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    return np.column_stack([-tangent[:, 1], tangent[:, 0]])


def _self_intersects(points: FloatArray) -> bool:
    """True se o polígono fechado cruza a si mesmo (ignora segmentos vizinhos)."""
    segs = polyline_to_segments(points)
    n = len(segs)
    hits = segments_intersect(segs, segs)
    idx = np.arange(n)
    dist = np.abs(idx[:, None] - idx[None, :])
    neighbours = (dist <= 1) | (dist >= n - 1)
    return bool(np.any(hits & ~neighbours))


def _folds_over(border: FloatArray, center: FloatArray, half_width: float) -> bool:
    """True se algum ponto da borda ficou mais perto da linha central do que meia largura.

    Numa curva fechada demais, o deslocamento "dobra" e a borda invade o outro
    lado da pista; isso aparece como um ponto perto demais de *outro* trecho
    da linha central, mesmo quando a borda não chega a se cruzar.
    """
    dist = np.linalg.norm(border[:, None, :] - center[None, :, :], axis=-1)
    return bool(np.min(dist) < 0.95 * half_width)


def build_track(
    name: str,
    control_points: FloatArray,
    width: float,
    smoothing: int = 3,
    spacing: float = 20.0,
    checkpoint_every: int = 3,
) -> Track:
    """Gera uma pista a partir da linha central de controle.

    Raises:
        ValueError: se alguma borda se auto-intersecta (curva fechada demais
            para a largura escolhida) ou se as bordas se cruzam.
    """
    center = resample_closed(chaikin_smooth(np.asarray(control_points, float), smoothing), spacing)
    offset = _normals(center) * (width / 2.0)
    left, right = center + offset, center - offset
    inner, outer = (
        (left, right) if abs(signed_area(left)) < abs(signed_area(right)) else (right, left)
    )

    for label, border in (("interna", inner), ("externa", outer)):
        if _self_intersects(border) or _folds_over(border, center, width / 2.0):
            raise ValueError(
                f"Pista '{name}': borda {label} se auto-intersecta ou dobra; "
                "aumente o raio das curvas ou reduza a largura."
            )
    if np.any(segments_intersect(polyline_to_segments(inner), polyline_to_segments(outer))):
        raise ValueError(f"Pista '{name}': bordas interna e externa se cruzam.")

    cp_idx = np.arange(0, len(center), checkpoint_every)
    checkpoints = np.hstack([left[cp_idx], right[cp_idx]])
    heading_vec = center[1] - center[0]
    return Track(
        name=name,
        width=width,
        centerline=center,
        inner=inner,
        outer=outer,
        walls=np.vstack([polyline_to_segments(inner), polyline_to_segments(outer)]),
        checkpoints=checkpoints,
        start_pos=center[0].copy(),
        start_heading=float(np.arctan2(heading_vec[1], heading_vec[0])),
    )


def track_from_dict(data: dict[str, Any]) -> Track:
    """Constrói a pista a partir do dicionário lido do YAML."""
    return build_track(
        name=str(data["name"]),
        control_points=np.asarray(data["control_points"], dtype=np.float64),
        width=float(data["width"]),
        smoothing=int(data.get("smoothing", 3)),
        spacing=float(data.get("spacing", 20.0)),
        checkpoint_every=int(data.get("checkpoint_every", 3)),
    )


def load_track(path: str | Path) -> Track:
    """Lê uma pista de um arquivo YAML."""
    with open(path, encoding="utf-8") as fh:
        return track_from_dict(yaml.safe_load(fh))
