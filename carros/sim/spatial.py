"""Grade espacial de paredes (*broad phase*).

Problema: testar cada carro contra todas as paredes custa ``N x W`` por passo.
Solução clássica de jogos: dividir o mapa em células e pré-calcular, para cada
célula, quais paredes *podem* interessar a um carro dentro dela. Em tempo de
execução basta descobrir a célula do carro e testar só essas candidatas
(*narrow phase*). O resultado é idêntico ao da força bruta, só que mais barato.

Para manter tudo vetorizado, as listas de candidatas são "acolchoadas" até o
mesmo tamanho ``K`` com uma parede nula (comprimento zero, nunca é atingida).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from carros.sim.geometry import FloatArray

IntArray = NDArray[np.int64]

_NULL_WALL = np.array([[1e9, 1e9, 1e9, 1e9]])


@dataclass(frozen=True)
class WallGrid:
    """Índice de paredes por célula.

    Attributes:
        walls: ``(W+1, 4)`` paredes; a última é a parede nula de preenchimento.
        near: ``(G, K1)`` candidatas para colisão (paredes a até ``near_margin``).
        far: ``(G, K2)`` candidatas para os raios (paredes a até ``ray_range``).
    """

    origin: FloatArray
    cell_size: float
    shape: tuple[int, int]
    walls: FloatArray
    near: IntArray
    far: IntArray

    def cells(self, points: FloatArray) -> IntArray:
        """Índice linear da célula de cada ponto ``(N, 2)`` (fora da grade = borda)."""
        ij = np.floor((points - self.origin) / self.cell_size).astype(np.int64)
        i = np.clip(ij[:, 0], 0, self.shape[0] - 1)
        j = np.clip(ij[:, 1], 0, self.shape[1] - 1)
        result: IntArray = i * self.shape[1] + j
        return result

    def near_walls(self, points: FloatArray) -> FloatArray:
        """``(N, K1, 4)`` paredes candidatas a colidir com carros nesses pontos."""
        result: FloatArray = self.walls[self.near[self.cells(points)]]
        return result

    def far_walls(self, points: FloatArray) -> FloatArray:
        """``(N, K2, 4)`` paredes candidatas a serem atingidas pelos raios."""
        result: FloatArray = self.walls[self.far[self.cells(points)]]
        return result


def _candidates(walls: FloatArray, lo: FloatArray, hi: FloatArray, margin: float) -> IntArray:
    """Matriz ``(G, K)`` de paredes cuja caixa envolvente toca cada célula expandida."""
    wmin = np.minimum(walls[:, :2], walls[:, 2:])  # (W, 2)
    wmax = np.maximum(walls[:, :2], walls[:, 2:])
    overlap = np.all(
        (wmin[None] <= hi[:, None] + margin) & (wmax[None] >= lo[:, None] - margin), axis=-1
    )  # (G, W)
    k = max(1, int(np.max(overlap.sum(axis=1))))
    null_idx = len(walls)
    out = np.full((len(lo), k), null_idx, dtype=np.int64)
    for g, row in enumerate(overlap):
        idx = np.flatnonzero(row)
        out[g, : idx.size] = idx
    return out


def build_wall_grid(
    walls: FloatArray, cell_size: float, near_margin: float, ray_range: float
) -> WallGrid:
    """Constrói a grade cobrindo as paredes (com folga de uma célula em volta).

    A margem é conservadora (caixa envolvente expandida), então nunca descarta
    uma parede que a força bruta encontraria.
    """
    lo_all = np.minimum(walls[:, :2], walls[:, 2:]).min(axis=0) - cell_size
    hi_all = np.maximum(walls[:, :2], walls[:, 2:]).max(axis=0) + cell_size
    shape = tuple(int(v) for v in np.ceil((hi_all - lo_all) / cell_size))
    gi, gj = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    lo = lo_all + np.column_stack([gi.ravel(), gj.ravel()]) * cell_size
    hi = lo + cell_size
    return WallGrid(
        origin=lo_all,
        cell_size=cell_size,
        shape=(shape[0], shape[1]),
        walls=np.vstack([walls, _NULL_WALL]),
        near=_candidates(walls, lo, hi, near_margin),
        far=_candidates(walls, lo, hi, ray_range),
    )
