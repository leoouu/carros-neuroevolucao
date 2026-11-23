"""Uma geração do algoritmo genético: elitismo + torneio + crossover + mutação."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from carros.evolution.config import GAParams
from carros.evolution.operators import (
    diversity,
    gaussian_mutation,
    single_point_crossover,
    tournament_select,
    uniform_crossover,
)
from carros.sim.geometry import FloatArray


@dataclass(frozen=True)
class Offspring:
    """Resultado da reprodução, com o que aconteceu para registrar no histórico."""

    genomes: FloatArray
    sigma: float
    diversity: float
    injected: int


def mutation_sigma(params: GAParams, generation: int) -> float:
    """Intensidade da mutação com decaimento: passos grandes no início, ajuste fino depois."""
    return max(params.sigma_min, params.mutation_sigma * params.sigma_decay**generation)


def next_generation(
    genomes: FloatArray,
    fitness: FloatArray,
    params: GAParams,
    generation: int,
    init_scale: float,
    rng: np.random.Generator,
) -> Offspring:
    """Produz a população da próxima geração.

    1. **Elitismo:** os ``elite`` melhores passam intactos (o melhor nunca piora).
    2. **Diversidade:** se a população ficou parecida demais (convergência
       prematura), a mutação é multiplicada por ``diversity_boost`` e uma fração
       vira indivíduos aleatórios novos.
    3. **Reprodução:** pais por torneio, crossover com prob. ``crossover_rate``
       (senão o filho é cópia do pai ``a``), depois mutação gaussiana.
    """
    pop = len(genomes)
    order = np.argsort(-fitness, kind="stable")
    elites = genomes[order[: params.elite]]

    div = diversity(genomes)
    sigma = mutation_sigma(params, generation)
    injected = 0
    if div < params.diversity_threshold:
        sigma *= params.diversity_boost
        injected = int(round(params.inject_fraction * pop))
    n_children = pop - params.elite - injected

    a = genomes[tournament_select(fitness, n_children, params.tournament_size, rng)]
    b = genomes[tournament_select(fitness, n_children, params.tournament_size, rng)]
    cross = uniform_crossover if params.crossover == "uniform" else single_point_crossover
    children = cross(a, b, rng)
    no_cross = rng.random(n_children) >= params.crossover_rate
    children[no_cross] = a[no_cross]
    children = gaussian_mutation(children, params.mutation_rate, sigma, rng)

    fresh = rng.normal(0.0, init_scale, size=(injected, genomes.shape[1]))
    new = np.vstack([elites, children, fresh])
    return Offspring(genomes=new, sigma=sigma, diversity=div, injected=injected)
