"""Avaliação: roda episódios com redes (ou baselines) dirigindo, em uma ou várias pistas.

Um *controlador* é qualquer função ``World -> (volante, acelerador)``. A rede
neural, a heurística e o carro aleatório são todos controladores, então o mesmo
código mede a evolução e os baselines de forma idêntica.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import dataclass
from types import TracebackType

import numpy as np
from numpy.typing import NDArray

from carros.baselines import heuristic_policy, random_policy
from carros.brain.mlp import MLPSpec, forward
from carros.evolution.config import EpisodeParams, FitnessParams
from carros.evolution.fitness import compute_fitness
from carros.sim.config import SimConfig, load_sim_config
from carros.sim.geometry import BoolArray, FloatArray
from carros.sim.sensors import ray_offsets
from carros.sim.track import Track, load_track
from carros.sim.world import Status, World

Controller = Callable[[World], tuple[FloatArray, FloatArray]]


@dataclass(frozen=True)
class EvalResult:
    """Estatísticas por carro ao fim do episódio (arrays ``(P,)``)."""

    fitness: FloatArray
    progress: FloatArray  # fração da volta em [0, 1]
    finished: BoolArray
    time_alive: FloatArray
    lap_time: FloatArray  # NaN para quem não completou
    status: NDArray[np.int64]


# ----------------------------------------------------------------- controladores
def network_controller(spec: MLPSpec, genomes: FloatArray) -> Controller:
    """Cada carro ``i`` é dirigido pela rede ``genomes[i]`` (só os vivos são calculados)."""

    def control(world: World) -> tuple[FloatArray, FloatArray]:
        out = np.zeros((world.n_cars, 2))
        idx = np.flatnonzero(world.alive)
        if idx.size:
            out[idx] = forward(spec, genomes[idx], world.observe()[idx])
        return out[:, 0], out[:, 1]

    return control


def heuristic_controller(sim_cfg: SimConfig) -> Controller:
    """Baseline: vira para o lado com mais espaço."""
    offsets = ray_offsets(sim_cfg.sensors)
    return lambda world: heuristic_policy(world.sensors(), offsets)


def random_controller(seed: int) -> Controller:
    """Baseline: comandos aleatórios a cada passo."""
    rng = np.random.default_rng(seed)
    return lambda world: random_policy(world.n_cars, rng)


# ----------------------------------------------------------------- episódio
def run_episode(
    track: Track,
    sim_cfg: SimConfig,
    n_cars: int,
    controller: Controller,
    episode: EpisodeParams,
    fitness_params: FitnessParams,
) -> EvalResult:
    """Simula ``n_cars`` até todos pararem ou o tempo acabar.

    Os carros não interagem entre si (atravessam uns aos outros), então rodar
    100 juntos dá exatamente o mesmo resultado que rodar um por vez — só que
    vetorizado.
    """
    world = World(track, sim_cfg, n_cars=n_cars, target_laps=episode.target_laps)
    for _ in range(int(round(episode.max_time / sim_cfg.dt))):
        if not world.any_alive():
            break
        world.step(*controller(world))

    progress = world.progress() / track.n_checkpoints
    finished: BoolArray = world.status == Status.FINISHED
    return EvalResult(
        fitness=compute_fitness(progress, finished, world.time_alive, fitness_params),
        progress=progress,
        finished=finished,
        time_alive=world.time_alive.copy(),
        lap_time=world.first_lap_time.copy(),
        status=world.status.copy(),
    )


def evaluate(
    genomes: FloatArray,
    spec: MLPSpec,
    track: Track,
    sim_cfg: SimConfig,
    episode: EpisodeParams,
    fitness_params: FitnessParams,
) -> EvalResult:
    """Avalia uma população de redes numa pista."""
    controller = network_controller(spec, genomes)
    return run_episode(track, sim_cfg, len(genomes), controller, episode, fitness_params)


# ----------------------------------------------------------------- várias pistas
class TrackSet:
    """Pistas avaliadas juntas; em paralelo (uma tarefa por pista) se ``workers > 1``.

    Paralelizar *entre pistas* é o corte natural: cada pista é independente e
    já vetoriza a população inteira, então não há nada a sincronizar.
    """

    def __init__(self, track_paths: Sequence[str], sim_path: str, workers: int = 1) -> None:
        self.tracks = [load_track(p) for p in track_paths]
        self.sim_cfg = load_sim_config(sim_path)
        self._executor: Executor | None = None
        if workers > 1:
            self._executor = ProcessPoolExecutor(max_workers=workers)

    def evaluate(
        self,
        genomes: FloatArray,
        spec: MLPSpec,
        episode: EpisodeParams,
        fitness_params: FitnessParams,
        extra: Sequence[Track] = (),
    ) -> list[EvalResult]:
        """Um ``EvalResult`` por pista: as fixas, na ordem, e depois as ``extra``."""
        jobs = [
            (genomes, spec, track, self.sim_cfg, episode, fitness_params)
            for track in [*self.tracks, *extra]
        ]
        if self._executor is None:
            return [evaluate(*job) for job in jobs]
        futures = [self._executor.submit(evaluate, *job) for job in jobs]
        return [f.result() for f in futures]

    def close(self) -> None:
        """Encerra os processos auxiliares."""
        if self._executor is not None:
            self._executor.shutdown()

    def __enter__(self) -> TrackSet:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def mean_fitness(results: Sequence[EvalResult]) -> FloatArray:
    """Fitness final de cada carro: média entre as pistas.

    Média (e não soma) para que o número tenha a mesma escala com 1 ou 10 pistas.
    """
    return np.mean([r.fitness for r in results], axis=0)
