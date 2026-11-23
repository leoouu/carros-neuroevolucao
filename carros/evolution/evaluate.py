"""Avaliação: roda um episódio com a população inteira dirigindo ao mesmo tempo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from carros.brain.mlp import MLPSpec, forward
from carros.evolution.config import EpisodeParams, FitnessParams
from carros.evolution.fitness import compute_fitness
from carros.sim.config import SimConfig
from carros.sim.geometry import BoolArray, FloatArray
from carros.sim.track import Track
from carros.sim.world import Status, World


@dataclass(frozen=True)
class EvalResult:
    """Estatísticas por carro ao fim do episódio (arrays ``(P,)``)."""

    fitness: FloatArray
    progress: FloatArray
    finished: BoolArray
    time_alive: FloatArray
    lap_time: FloatArray  # NaN para quem não completou
    status: NDArray[np.int64]


def drive(world: World, spec: MLPSpec, genomes: FloatArray) -> None:
    """Um passo: cada carro vivo pergunta à sua rede o que fazer."""
    controls = np.zeros((world.n_cars, 2))
    idx = np.flatnonzero(world.alive)
    if idx.size:
        controls[idx] = forward(spec, genomes[idx], world.observe()[idx])
    world.step(controls[:, 0], controls[:, 1])


def evaluate(
    genomes: FloatArray,
    spec: MLPSpec,
    track: Track,
    sim_cfg: SimConfig,
    episode: EpisodeParams,
    fitness_params: FitnessParams,
) -> EvalResult:
    """Simula todos os genomas juntos até todos pararem ou o tempo acabar.

    Os carros não interagem entre si (atravessam uns aos outros), então rodar
    100 juntos dá exatamente o mesmo resultado que rodar um por vez — só que
    vetorizado.
    """
    world = World(track, sim_cfg, n_cars=len(genomes), target_laps=episode.target_laps)
    for _ in range(int(round(episode.max_time / sim_cfg.dt))):
        if not world.any_alive():
            break
        drive(world, spec, genomes)

    progress = world.progress()
    finished: BoolArray = world.status == Status.FINISHED
    return EvalResult(
        fitness=compute_fitness(progress, finished, world.time_alive, fitness_params),
        progress=progress,
        finished=finished,
        time_alive=world.time_alive.copy(),
        lap_time=world.first_lap_time.copy(),
        status=world.status.copy(),
    )
