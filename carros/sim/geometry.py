"""Primitivas geométricas vetorizadas (NumPy).

Convenção: um segmento é uma linha de 4 números ``[x1, y1, x2, y2]``.
Todas as funções trabalham em lote para que 100 carros x 7 raios x centenas de
paredes sejam resolvidos com poucas operações de array, sem laços Python.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

_EPS = 1e-12


def _cross(ax: FloatArray, ay: FloatArray, bx: FloatArray, by: FloatArray) -> FloatArray:
    """Produto vetorial 2D (componente z) elemento a elemento."""
    return ax * by - ay * bx


def ray_segment_distances(
    origins: FloatArray, directions: FloatArray, walls: FloatArray, max_range: float
) -> FloatArray:
    """Distância de cada raio até a parede mais próxima.

    Args:
        origins: ``(N, 2)`` origem de cada carro.
        directions: ``(N, R, 2)`` vetores unitários dos ``R`` raios de cada carro.
        walls: ``(W, 4)`` paredes comuns a todos, ou ``(N, K, 4)`` candidatas por carro
            (vindas da grade espacial).
        max_range: alcance máximo; raios sem colisão retornam esse valor.

    Returns:
        ``(N, R)`` distâncias em ``[0, max_range]``.

    Resolve ``p + t*d = a + u*(b - a)`` com ``t >= 0`` e ``0 <= u <= 1``.
    """
    w = walls[None] if walls.ndim == 2 else walls  # (N|1, K, 4)
    px = origins[:, 0][:, None, None]  # (N,1,1)
    py = origins[:, 1][:, None, None]
    dx = directions[..., 0][..., None]  # (N,R,1)
    dy = directions[..., 1][..., None]
    ax, ay = w[:, None, :, 0], w[:, None, :, 1]  # (N|1,1,K)
    ex, ey = w[:, None, :, 2] - ax, w[:, None, :, 3] - ay

    denom = _cross(dx, dy, ex, ey)  # (N,R,K)
    apx, apy = ax - px, ay - py  # (N,1,K)
    parallel = np.abs(denom) < _EPS
    safe = np.where(parallel, 1.0, denom)
    t = _cross(apx, apy, ex, ey) / safe
    u = _cross(apx, apy, dx, dy) / safe

    hit = (~parallel) & (t >= 0.0) & (u >= 0.0) & (u <= 1.0)
    t = np.where(hit, t, max_range)
    return np.minimum(t.min(axis=-1), max_range)


def _intersect(a: FloatArray, b: FloatArray) -> BoolArray:
    """Núcleo com broadcasting: ``a`` e ``b`` têm forma ``(..., 4)`` compatível."""
    px, py = a[..., 0], a[..., 1]
    rx, ry = a[..., 2] - px, a[..., 3] - py
    qx, qy = b[..., 0], b[..., 1]
    sx, sy = b[..., 2] - qx, b[..., 3] - qy

    denom = _cross(rx, ry, sx, sy)
    parallel = np.abs(denom) < _EPS
    safe = np.where(parallel, 1.0, denom)
    qpx, qpy = qx - px, qy - py
    t = _cross(qpx, qpy, sx, sy) / safe
    u = _cross(qpx, qpy, rx, ry) / safe
    result: BoolArray = (~parallel) & (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)
    return result


def segments_intersect(segs_a: FloatArray, segs_b: FloatArray) -> BoolArray:
    """Matriz ``(A, B)`` indicando quais pares de segmentos se cruzam.

    Segmentos colineares sobrepostos são tratados como não-intersectantes;
    para colisão de carro isso é irrelevante (probabilidade zero em float).
    """
    return _intersect(segs_a[:, None, :], segs_b[None, :, :])


def segments_intersect_pairwise(segs_a: FloatArray, segs_b: FloatArray) -> BoolArray:
    """``segs_a[i]`` cruza ``segs_b[i]``? Formas ``(..., 4)`` compatíveis por broadcasting."""
    return _intersect(segs_a, segs_b)


def polyline_to_segments(points: FloatArray, closed: bool = True) -> FloatArray:
    """Converte ``(P, 2)`` pontos em ``(P, 4)`` segmentos (ou ``P-1`` se aberto)."""
    end = np.roll(points, -1, axis=0) if closed else points[1:]
    start = points if closed else points[:-1]
    return np.hstack([start, end]).astype(np.float64)


def chaikin_smooth(points: FloatArray, iterations: int) -> FloatArray:
    """Suaviza um polígono fechado pelo algoritmo de Chaikin (corta os cantos)."""
    pts = points.astype(np.float64)
    for _ in range(iterations):
        nxt = np.roll(pts, -1, axis=0)
        q = 0.75 * pts + 0.25 * nxt
        r = 0.25 * pts + 0.75 * nxt
        pts = np.empty((2 * len(q), 2))
        pts[0::2], pts[1::2] = q, r
    return pts


def resample_closed(points: FloatArray, spacing: float) -> FloatArray:
    """Reamostra um polígono fechado com pontos igualmente espaçados (aprox. ``spacing``)."""
    closed = np.vstack([points, points[:1]])
    seg_len = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(cum[-1])
    n = max(3, int(round(total / spacing)))
    targets = np.linspace(0.0, total, n, endpoint=False)
    x = np.interp(targets, cum, closed[:, 0])
    y = np.interp(targets, cum, closed[:, 1])
    return np.column_stack([x, y])


def signed_area(points: FloatArray) -> float:
    """Área com sinal (fórmula do laço). Positiva = anti-horário em eixos matemáticos."""
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
