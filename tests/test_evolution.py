"""Fitness, operadores genéticos, uma geração da GA e avaliação."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from carros.brain.mlp import MLPSpec, random_genomes
from carros.evolution.config import EpisodeParams, FitnessParams, GAParams
from carros.evolution.evaluate import evaluate
from carros.evolution.fitness import compute_fitness
from carros.evolution.ga import mutation_sigma, next_generation
from carros.evolution.operators import (
    diversity,
    gaussian_mutation,
    single_point_crossover,
    tournament_select,
    uniform_crossover,
)
from carros.sim.config import SimConfig
from carros.sim.track import Track

FIT = FitnessParams(time_penalty=0.02, lap_bonus=50.0)
GA = GAParams(
    population=20,
    elite=2,
    tournament_size=3,
    crossover="uniform",
    crossover_rate=0.8,
    mutation_rate=0.1,
    mutation_sigma=0.4,
    sigma_decay=0.9,
    sigma_min=0.05,
    diversity_threshold=0.1,
    diversity_boost=3.0,
    inject_fraction=0.25,
)


# ------------------------------------------------------------------ fitness
def test_fitness_formula() -> None:
    f = compute_fitness(np.array([3.5, 45.0]), np.array([False, True]), np.array([10.0, 8.0]), FIT)
    np.testing.assert_allclose(f, [3.5 - 0.2, 45.0 + 50.0 - 0.16])


def test_more_progress_beats_less_progress() -> None:
    f = compute_fitness(np.array([2.0, 2.6]), np.zeros(2, bool), np.array([1.0, 1.0]), FIT)
    assert f[1] > f[0]


def test_finishing_beats_any_unfinished_run() -> None:
    n_cps = 45
    unfinished = compute_fitness(np.array([n_cps - 0.01]), np.zeros(1, bool), np.zeros(1), FIT)
    finished = compute_fitness(np.array([float(n_cps)]), np.ones(1, bool), np.array([40.0]), FIT)
    assert finished[0] > unfinished[0]


def test_faster_lap_beats_slower_lap() -> None:
    f = compute_fitness(np.array([45.0, 45.0]), np.ones(2, bool), np.array([8.0, 12.0]), FIT)
    assert f[0] > f[1]


def test_crashing_early_does_not_pay_off() -> None:
    """A penalidade de tempo é leve: andar 1 checkpoint a mais compensa 10 s vivo."""
    early = compute_fitness(np.array([1.0]), np.zeros(1, bool), np.array([1.0]), FIT)
    later = compute_fitness(np.array([2.0]), np.zeros(1, bool), np.array([11.0]), FIT)
    assert later[0] > early[0]


# ------------------------------------------------------------------ operadores
def test_tournament_prefers_fitter_individuals() -> None:
    rng = np.random.default_rng(0)
    fitness = np.arange(10, dtype=float)
    picks = tournament_select(fitness, 5000, 3, rng)
    assert picks.mean() > 4.5 + 1.5  # bem acima da média uniforme (4.5)
    assert np.all(tournament_select(fitness, 50, 10_000, rng) == 9)


def test_uniform_crossover_takes_each_gene_from_a_parent() -> None:
    rng = np.random.default_rng(1)
    a, b = np.zeros((50, 30)), np.ones((50, 30))
    child = uniform_crossover(a, b, rng)
    assert set(np.unique(child)) <= {0.0, 1.0}
    assert 0.4 < child.mean() < 0.6


def test_single_point_crossover_is_prefix_of_a_and_suffix_of_b() -> None:
    rng = np.random.default_rng(2)
    a, b = np.zeros((40, 10)), np.ones((40, 10))
    child = single_point_crossover(a, b, rng)
    for row in child:
        cut = int(np.argmax(row))  # primeiro gene vindo de b
        assert 1 <= cut <= 9
        assert np.all(row[:cut] == 0) and np.all(row[cut:] == 1)


def test_mutation_rate_and_sigma() -> None:
    rng = np.random.default_rng(3)
    g = np.zeros((200, 100))
    m = gaussian_mutation(g, rate=0.1, sigma=0.5, rng=rng)
    changed = m != 0
    assert 0.08 < changed.mean() < 0.12
    assert m[changed].std() == pytest.approx(0.5, rel=0.1)


def test_mutation_with_zero_rate_is_identity() -> None:
    g = np.random.default_rng(4).normal(size=(5, 7))
    np.testing.assert_array_equal(gaussian_mutation(g, 0.0, 1.0, np.random.default_rng(5)), g)


def test_diversity_zero_for_clones() -> None:
    assert diversity(np.tile(np.arange(5.0), (10, 1))) == 0.0
    assert diversity(np.random.default_rng(6).normal(size=(500, 5))) == pytest.approx(1.0, 0.1)


# ------------------------------------------------------------------ geração
def test_sigma_decays_to_floor() -> None:
    assert mutation_sigma(GA, 0) == pytest.approx(0.4)
    assert mutation_sigma(GA, 1) == pytest.approx(0.36)
    assert mutation_sigma(GA, 1000) == GA.sigma_min


def test_next_generation_keeps_elites_and_size() -> None:
    rng = np.random.default_rng(7)
    genomes = rng.normal(size=(20, 12))
    fitness = rng.random(20)
    off = next_generation(genomes, fitness, GA, 0, 1.0, rng)
    assert off.genomes.shape == genomes.shape
    best_two = genomes[np.argsort(-fitness)[:2]]
    np.testing.assert_array_equal(off.genomes[:2], best_two)
    assert off.injected == 0


def test_low_diversity_triggers_boost_and_injection() -> None:
    rng = np.random.default_rng(8)
    clones = np.tile(rng.normal(size=12), (20, 1))
    off = next_generation(clones, np.zeros(20), GA, 0, 1.0, rng)
    assert off.injected == 5
    assert off.sigma == pytest.approx(0.4 * 3.0)
    assert diversity(off.genomes) > diversity(clones)


def test_next_generation_is_reproducible() -> None:
    genomes = np.random.default_rng(9).normal(size=(20, 12))
    fitness = np.random.default_rng(10).random(20)
    a = next_generation(genomes, fitness, GA, 3, 1.0, np.random.default_rng(11)).genomes
    b = next_generation(genomes, fitness, GA, 3, 1.0, np.random.default_rng(11)).genomes
    np.testing.assert_array_equal(a, b)


# ------------------------------------------------------------------ avaliação
def test_evaluate_shapes_and_consistency(cfg: SimConfig, simple_track: Track) -> None:
    spec = MLPSpec((cfg.sensors.count + 1, 4, 2))
    genomes = random_genomes(spec, 12, np.random.default_rng(12), 1.0)
    res = evaluate(genomes, spec, simple_track, cfg, EpisodeParams(5.0, 1), FIT)
    assert res.fitness.shape == (12,)
    np.testing.assert_allclose(
        res.fitness, compute_fitness(res.progress, res.finished, res.time_alive, FIT)
    )
    assert np.all(res.time_alive <= 5.0 + 1e-9)


def test_evaluate_is_deterministic(cfg: SimConfig, simple_track: Track) -> None:
    spec = MLPSpec((cfg.sensors.count + 1, 4, 2))
    genomes = random_genomes(spec, 12, np.random.default_rng(13), 1.0)
    ep = EpisodeParams(3.0, 1)
    a = evaluate(genomes, spec, simple_track, cfg, ep, FIT)
    b = evaluate(genomes, spec, simple_track, cfg, ep, FIT)
    np.testing.assert_array_equal(a.fitness, b.fitness)


def test_world_progress_and_finish(cfg: SimConfig, simple_track: Track) -> None:
    """Heurística com target_laps=1: termina com progresso = nº de checkpoints."""
    from carros.baselines import heuristic_policy
    from carros.sim.sensors import ray_offsets
    from carros.sim.world import Status, World

    world = World(simple_track, dataclasses.replace(cfg), target_laps=1)
    off = ray_offsets(cfg.sensors)
    p0 = world.progress()[0]
    world.step(*heuristic_policy(world.sensors(), off))
    assert world.progress()[0] > p0  # andar para frente aumenta o progresso
    while world.any_alive() and world.time < 60:
        world.step(*heuristic_policy(world.sensors(), off))
    assert world.status[0] == Status.FINISHED
    assert world.progress()[0] == simple_track.n_checkpoints
    assert world.time_alive[0] == pytest.approx(world.first_lap_time[0])
