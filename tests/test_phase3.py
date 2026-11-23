"""Gerador de pistas, pistas feitas à mão, avaliação em várias pistas e config de treino."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from carros.brain.mlp import MLPSpec, random_genomes
from carros.evolution.config import (
    EpisodeParams,
    FitnessParams,
    expand_track_paths,
    load_train_config,
    train_config_from_dict,
)
from carros.evolution.evaluate import (
    TrackSet,
    evaluate,
    heuristic_controller,
    mean_fitness,
    run_episode,
)
from carros.evolution.trainer import summarize
from carros.sim.config import load_sim_config
from carros.sim.generator import (
    GeneratorParams,
    generate_track,
    generate_track_dict,
    load_generator_params,
)
from carros.sim.geometry import signed_area
from carros.sim.track import load_track, track_from_dict

FIT = FitnessParams(time_penalty=0.02, lap_bonus=50.0)
HAND_MADE = ["simple", "hairpin", "s_curves", "chicane"]


@pytest.fixture(scope="module")
def gen_params() -> GeneratorParams:
    """Parâmetros reais do gerador."""
    return load_generator_params("configs/generator.yaml")


# ------------------------------------------------------------------ gerador
def test_generator_is_deterministic(gen_params: GeneratorParams) -> None:
    assert generate_track_dict("a", gen_params, 7) == generate_track_dict("a", gen_params, 7)
    assert generate_track_dict("a", gen_params, 7) != generate_track_dict("a", gen_params, 8)


def test_generated_tracks_are_valid_and_vary_direction(gen_params: GeneratorParams) -> None:
    orientations = set()
    for seed in range(12):
        track = generate_track(f"t{seed}", gen_params, seed)  # build_track já valida
        assert track.n_checkpoints >= 10
        orientations.add(float(np.sign(signed_area(track.centerline))))
    assert orientations == {-1.0, 1.0}  # há pistas nos dois sentidos de corrida


def test_saved_generated_tracks_match_generator(gen_params: GeneratorParams) -> None:
    """Os YAMLs versionados em configs/tracks/train ainda saem do gerador atual."""
    saved = yaml.safe_load(Path("configs/tracks/train/train_0003.yaml").read_text("utf-8"))
    assert saved == generate_track_dict("train_0003", gen_params, 3)


def test_track_dict_roundtrip_through_yaml(gen_params: GeneratorParams) -> None:
    data = generate_track_dict("x", gen_params, 42)
    again = yaml.safe_load(yaml.safe_dump(data))
    np.testing.assert_allclose(track_from_dict(again).walls, track_from_dict(data).walls)


# ------------------------------------------------------------------ pistas à mão
@pytest.mark.parametrize("name", HAND_MADE)
def test_heuristic_completes_each_hand_made_track(name: str) -> None:
    track = load_track(f"configs/tracks/{name}.yaml")
    sim = load_sim_config("configs/sim_default.yaml")
    res = run_episode(track, sim, 1, heuristic_controller(sim), EpisodeParams(40.0, 1), FIT)
    assert res.finished[0]
    assert res.progress[0] == pytest.approx(1.0)  # progresso é fração da volta


def test_train_and_unseen_tracks_are_disjoint() -> None:
    train = {Path(p).read_text("utf-8") for p in expand_track_paths(["configs/tracks/train"])}
    unseen = {Path(p).read_text("utf-8") for p in expand_track_paths(["configs/tracks/unseen"])}
    assert train and unseen and not train & unseen


# ------------------------------------------------------------------ várias pistas
def test_parallel_track_set_matches_sequential() -> None:
    paths = ["configs/tracks/simple.yaml", "configs/tracks/hairpin.yaml"]
    sim_path = "configs/sim_default.yaml"
    spec = MLPSpec((8, 4, 2))
    genomes = random_genomes(spec, 10, np.random.default_rng(0), 1.0)
    ep = EpisodeParams(3.0, 1)
    with TrackSet(paths, sim_path, workers=2) as par, TrackSet(paths, sim_path, workers=1) as seq:
        a = par.evaluate(genomes, spec, ep, FIT)
        b = seq.evaluate(genomes, spec, ep, FIT)
    for ra, rb in zip(a, b, strict=True):
        np.testing.assert_array_equal(ra.fitness, rb.fitness)


def test_mean_fitness_and_summary() -> None:
    sim = load_sim_config("configs/sim_default.yaml")
    spec = MLPSpec((8, 4, 2))
    genomes = random_genomes(spec, 5, np.random.default_rng(1), 1.0)
    tracks = [load_track(f"configs/tracks/{n}.yaml") for n in ("simple", "chicane")]
    results = [evaluate(genomes, spec, t, sim, EpisodeParams(2.0, 1), FIT) for t in tracks]
    np.testing.assert_allclose(mean_fitness(results), (results[0].fitness + results[1].fitness) / 2)
    progress, done, lap = summarize(results, 0)
    assert progress == pytest.approx(50 * (results[0].progress[0] + results[1].progress[0]))
    assert done == 0.0 and np.isnan(lap)  # 2 s não dá para completar a volta


# ------------------------------------------------------------------ config
def test_phase_configs_load() -> None:
    cfg2, _ = load_train_config("configs/phase2_simple_track.yaml")
    cfg3, _ = load_train_config("configs/phase3_generalization.yaml")
    assert cfg2.tracks == ["configs/tracks/simple.yaml"]
    assert len(cfg3.tracks) == 11 and len(cfg3.validation_tracks) == 9


def test_validation_track_in_training_set_is_rejected() -> None:
    _, raw = load_train_config("configs/phase3_generalization.yaml")
    bad = copy.deepcopy(raw)
    bad["validation_tracks"] = ["configs/tracks/hairpin.yaml"]
    with pytest.raises(ValueError, match="validação"):
        train_config_from_dict(bad)
