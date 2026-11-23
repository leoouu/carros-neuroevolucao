"""Rede neural (MLP) que avalia uma população inteira de genomas de uma vez.

O **genoma** de um carro é o vetor plano com todos os pesos e vieses da rede,
camada por camada: ``[W1 (achatada), b1, W2 (achatada), b2, ...]``. Guardar assim
torna crossover e mutação triviais (operações em vetores), enquanto ``unflatten``
reconstrói as matrizes só na hora de calcular a saída.

Para ``P`` carros, as matrizes viram tensores ``(P, entrada, saída)`` e a passada
da rede é um ``einsum``: os 100 cérebros rodam numa única operação NumPy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from carros.sim.geometry import FloatArray


@dataclass(frozen=True)
class MLPSpec:
    """Arquitetura: tamanhos das camadas, da entrada à saída (ex.: ``(8, 8, 2)``)."""

    layer_sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.layer_sizes) < 2 or min(self.layer_sizes) < 1:
            raise ValueError(f"Arquitetura inválida: {self.layer_sizes}")

    @property
    def n_params(self) -> int:
        """Tamanho do genoma (pesos + vieses)."""
        pairs = zip(self.layer_sizes[:-1], self.layer_sizes[1:], strict=True)
        return sum(i * o + o for i, o in pairs)

    def unflatten(self, genomes: FloatArray) -> list[tuple[FloatArray, FloatArray]]:
        """Fatia ``(P, G)`` em ``[(W (P, i, o), b (P, o)), ...]`` por camada."""
        if genomes.shape[-1] != self.n_params:
            raise ValueError(f"Genoma com {genomes.shape[-1]} genes; esperado {self.n_params}")
        layers = []
        pos = 0
        pairs = zip(self.layer_sizes[:-1], self.layer_sizes[1:], strict=True)
        for i, o in pairs:
            w = genomes[:, pos : pos + i * o].reshape(-1, i, o)
            pos += i * o
            b = genomes[:, pos : pos + o]
            pos += o
            layers.append((w, b))
        return layers


def forward(spec: MLPSpec, genomes: FloatArray, inputs: FloatArray) -> FloatArray:
    """Saída ``(P, saídas)`` em ``[-1, 1]``: cada carro ``p`` usa o seu genoma ``p``.

    ``tanh`` em todas as camadas: nas ocultas dá não-linearidade; na saída já
    entrega volante e acelerador no intervalo ``[-1, 1]`` que a física espera.
    """
    h = inputs
    for w, b in spec.unflatten(genomes):
        h = np.tanh(np.einsum("pi,pio->po", h, w) + b)
    return h


def random_genomes(spec: MLPSpec, n: int, rng: np.random.Generator, scale: float) -> FloatArray:
    """``(n, G)`` genomas com genes ~ Normal(0, ``scale``)."""
    return rng.normal(0.0, scale, size=(n, spec.n_params))


def save_genome(path: str | Path, genome: FloatArray, spec: MLPSpec, meta: dict[str, Any]) -> None:
    """Salva ``genome`` em ``.npy`` e a arquitetura + metadados num ``.json`` ao lado."""
    path = Path(path).with_suffix(".npy")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(genome, dtype=np.float64))
    info = {"layer_sizes": list(spec.layer_sizes), **meta}
    path.with_suffix(".json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def load_genome(path: str | Path) -> tuple[FloatArray, MLPSpec, dict[str, Any]]:
    """Carrega genoma, arquitetura e metadados salvos por ``save_genome``."""
    path = Path(path).with_suffix(".npy")
    genome: FloatArray = np.load(path)
    meta: dict[str, Any] = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    spec = MLPSpec(tuple(int(v) for v in meta.pop("layer_sizes")))
    if genome.shape != (spec.n_params,):
        raise ValueError(f"{path}: genoma {genome.shape} incompatível com {spec}")
    return genome, spec, meta
