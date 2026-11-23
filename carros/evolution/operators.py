"""Operadores genéticos sobre genomas (vetores planos de pesos).

Todos recebem um ``np.random.Generator`` explícito: com a mesma semente, o
treino inteiro é reproduzível.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from carros.sim.geometry import FloatArray

IntArray = NDArray[np.int64]


def tournament_select(fitness: FloatArray, n: int, k: int, rng: np.random.Generator) -> IntArray:
    """Índices de ``n`` pais, cada um o melhor entre ``k`` sorteados.

    Por que torneio e não roleta? A roleta depende da *escala* do fitness (que
    aqui pode ser negativo e muda muito entre gerações). O torneio só olha a
    *ordem*, e ``k`` controla a pressão seletiva: k maior = só os melhores
    reproduzem (converge rápido, perde diversidade); k menor = mais exploração.
    """
    contestants = rng.integers(0, len(fitness), size=(n, k))
    winners = np.argmax(fitness[contestants], axis=1)
    result: IntArray = contestants[np.arange(n), winners]
    return result


def uniform_crossover(a: FloatArray, b: FloatArray, rng: np.random.Generator) -> FloatArray:
    """Cada gene vem de ``a`` ou de ``b`` com probabilidade 1/2."""
    mask = rng.random(a.shape) < 0.5
    result: FloatArray = np.where(mask, a, b)
    return result


def single_point_crossover(a: FloatArray, b: FloatArray, rng: np.random.Generator) -> FloatArray:
    """Genes ``[0, corte)`` de ``a`` e ``[corte, G)`` de ``b`` (um corte por filho).

    Preserva blocos contíguos (ex.: os pesos de um mesmo neurônio ficam juntos).
    """
    genes = a.shape[1]
    cut = rng.integers(1, genes, size=(len(a), 1))
    mask = np.arange(genes)[None, :] < cut
    result: FloatArray = np.where(mask, a, b)
    return result


def gaussian_mutation(
    genomes: FloatArray, rate: float, sigma: float, rng: np.random.Generator
) -> FloatArray:
    """Soma ruído Normal(0, ``sigma``) a uma fração ``rate`` dos genes.

    ``rate`` = quantos genes mudam; ``sigma`` = o tamanho do passo. Mutação alta
    demais vira busca aleatória; baixa demais e a população estagna.
    """
    mask = rng.random(genomes.shape) < rate
    noise = rng.normal(0.0, sigma, size=genomes.shape)
    result: FloatArray = genomes + mask * noise
    return result


def diversity(genomes: FloatArray) -> float:
    """Desvio padrão médio de cada gene na população (0 = todos idênticos)."""
    return float(np.mean(np.std(genomes, axis=0)))
