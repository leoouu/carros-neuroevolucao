"""Laço de treino: avalia, registra, salva o melhor e gera a próxima população."""

from __future__ import annotations

import csv
import json
import shutil
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from carros.brain.mlp import MLPSpec, random_genomes, save_genome
from carros.evolution.config import TrainConfig
from carros.evolution.evaluate import EvalResult, TrackSet, mean_fitness
from carros.evolution.ga import mutation_sigma, next_generation
from carros.evolution.operators import diversity
from carros.sim.config import SimConfig
from carros.sim.generator import generate_track, load_generator_params
from carros.sim.track import Track

NAN = float("nan")


@dataclass(frozen=True)
class GenerationRecord:
    """Uma linha do ``history.csv``. Percentuais em [0, 100]; NaN = não medido."""

    generation: int
    best_fitness: float
    mean_fitness: float
    median_fitness: float
    best_progress_pct: float  # % da volta do melhor carro (média das pistas de treino)
    finished_pct: float  # % dos pares (carro, pista) que completaram
    best_finished_pct: float  # % das pistas de treino que o melhor carro completou
    best_lap_time: float  # tempo médio de volta do melhor, se completou todas
    val_progress_pct: float  # melhor carro em pistas nunca vistas
    val_finished_pct: float
    diversity: float
    sigma: float
    injected: int
    seconds: float


def network_spec(cfg: TrainConfig, sim_cfg: SimConfig) -> MLPSpec:
    """Entradas = sensores + velocidade; saídas = volante e acelerador."""
    return MLPSpec((sim_cfg.sensors.count + 1, *cfg.network.hidden, 2))


def _save_run_config(out: Path, cfg: TrainConfig, raw: dict[str, Any]) -> None:
    """Guarda a config do experimento e cópias das pistas e da física usadas."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    shutil.copy(cfg.sim, out / "sim.yaml")
    for folder, paths in (("tracks", cfg.tracks), ("validation_tracks", cfg.validation_tracks)):
        shutil.rmtree(out / folder, ignore_errors=True)
        (out / folder).mkdir()
        for i, path in enumerate(paths):
            shutil.copy(path, out / folder / f"{i:02d}_{Path(path).name}")


def summarize(results: Sequence[EvalResult], car: int) -> tuple[float, float, float]:
    """Para o carro ``car``: (% médio da volta, % de pistas completadas, tempo médio de volta)."""
    progress = 100.0 * float(np.mean([r.progress[car] for r in results]))
    done = [bool(r.finished[car]) for r in results]
    finished = 100.0 * float(np.mean(done))
    lap = float(np.mean([r.lap_time[car] for r in results])) if all(done) else NAN
    return progress, finished, lap


def random_tracks(cfg: TrainConfig, generation: int) -> list[Track]:
    """Pistas inéditas da geração. A semente depende só da geração: reproduzível e retomável."""
    rt = cfg.random_tracks
    if rt.count == 0:
        return []
    params = load_generator_params(rt.generator)
    first = rt.base_seed + generation * rt.count
    return [generate_track(f"rnd_{s}", params, s) for s in range(first, first + rt.count)]


def population_path(run: Path, generation: int) -> Path:
    """Onde fica a população inteira de uma geração."""
    return run / "populations" / f"gen{generation:03d}.npy"


@dataclass(frozen=True)
class _State:
    """Tudo que é preciso para continuar um treino (além da população)."""

    generation: int
    sigma: float
    injected: int
    rng: dict[str, Any]


def _save_state(out: Path, state: _State, genomes: np.ndarray) -> None:
    """Grava a população e só então, atomicamente, o estado que aponta para ela."""
    path = population_path(out, state.generation)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, genomes)
    tmp = out / "state.tmp"
    tmp.write_text(json.dumps(asdict(state)), encoding="utf-8")
    tmp.replace(out / "state.json")


def _check_resume(out: Path, raw: dict[str, Any]) -> None:
    """Só retoma se a config for a mesma (``generations`` pode aumentar)."""
    if not (out / "state.json").exists():
        raise ValueError(f"nada para retomar em {out}")
    saved = yaml.safe_load((out / "config.yaml").read_text(encoding="utf-8"))
    if {**saved, "generations": 0} != {**raw, "generations": 0}:
        raise ValueError("a config mudou desde o treino salvo; não dá para retomar")


def _truncate_history(out: Path, before: int) -> list[GenerationRecord]:
    """Descarta registros a partir de ``before`` (geração que será refeita)."""
    history = [r for r in load_history(out / "history.csv") if r.generation < before]
    for i, rec in enumerate(history):
        _append_csv(out / "history.csv", rec, header=i == 0)
    return history


def train(
    cfg: TrainConfig,
    raw: dict[str, Any],
    log: Callable[[str], None] = print,
    resume: bool = False,
) -> list[GenerationRecord]:
    """Executa (ou retoma) o experimento e devolve o histórico por geração."""
    rng = np.random.default_rng(cfg.seed)
    out = Path(cfg.output_dir)
    if resume:
        _check_resume(out, raw)
        (out / "config.yaml").write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    else:
        _save_run_config(out, cfg, raw)
    with (
        TrackSet(cfg.tracks, cfg.sim, cfg.workers) as train_set,
        TrackSet(cfg.validation_tracks, cfg.sim, cfg.workers) as val_set,
    ):
        spec = network_spec(cfg, train_set.sim_cfg)
        log(
            f"Rede {spec.layer_sizes} -> {spec.n_params} genes; população {cfg.ga.population}; "
            f"{len(cfg.tracks)} pista(s) de treino, {len(cfg.validation_tracks)} de validação"
        )
        genomes = random_genomes(spec, cfg.ga.population, rng, cfg.network.init_scale)
        history: list[GenerationRecord] = []
        sigma, injected, start = mutation_sigma(cfg.ga, 0), 0, 1
        if resume:
            state = _State(**json.loads((out / "state.json").read_text(encoding="utf-8")))
            start, sigma, injected = state.generation, state.sigma, state.injected
            rng.bit_generator.state = state.rng
            genomes = np.load(population_path(out, start))
            if start > 1:
                history = _truncate_history(out, start)
            log(f"Retomando na geração {start}")
        for gen in range(start, cfg.generations + 1):
            t0 = time.perf_counter()
            _save_state(out, _State(gen, sigma, injected, dict(rng.bit_generator.state)), genomes)
            extra = random_tracks(cfg, gen)
            results = train_set.evaluate(genomes, spec, cfg.episode, cfg.fitness, extra)
            fitness = mean_fitness(results)
            best = int(np.argmax(fitness))
            progress, best_done, lap = summarize(results, best)

            val_progress = val_done = NAN
            checkpoint_gen = gen % cfg.validate_every == 0 or gen in (1, cfg.generations)
            if val_set.tracks and checkpoint_gen:
                # Só o melhor da geração (escolhido pelo TREINO) corre nas pistas novas:
                # escolher pelo resultado da validação seria "colar" na prova.
                val = val_set.evaluate(genomes[best : best + 1], spec, cfg.episode, cfg.fitness)
                val_progress, val_done, _ = summarize(val, 0)

            rec = GenerationRecord(
                generation=gen,
                best_fitness=float(fitness[best]),
                mean_fitness=float(np.mean(fitness)),
                median_fitness=float(np.median(fitness)),
                best_progress_pct=progress,
                finished_pct=100.0 * float(np.mean([r.finished for r in results])),
                best_finished_pct=best_done,
                best_lap_time=lap,
                val_progress_pct=val_progress,
                val_finished_pct=val_done,
                diversity=diversity(genomes),
                sigma=sigma,
                injected=injected,
                seconds=time.perf_counter() - t0,
            )
            history.append(rec)
            meta = {k: v for k, v in asdict(rec).items() if k not in ("seconds",)}
            save_genome(
                out / f"best_gen{gen:03d}",
                genomes[best],
                spec,
                {"experiment": cfg.name, **{k: (None if v != v else v) for k, v in meta.items()}},
            )
            _append_csv(out / "history.csv", rec, header=gen == 1)
            log(_format(rec))

            if gen < cfg.generations:
                off = next_generation(genomes, fitness, cfg.ga, gen, cfg.network.init_scale, rng)
                genomes, sigma, injected = off.genomes, off.sigma, off.injected
                if injected:
                    log(
                        f"  ! diversidade {off.diversity:.3f} baixa: mutação x"
                        f"{cfg.ga.diversity_boost}, {injected} aleatórios injetados"
                    )
    return history


def _append_csv(path: Path, rec: GenerationRecord, header: bool) -> None:
    row = asdict(rec)
    with open(path, "w" if header else "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row))
        if header:
            writer.writeheader()
        writer.writerow(row)


def _format(rec: GenerationRecord) -> str:
    lap = "  -   " if np.isnan(rec.best_lap_time) else f"{rec.best_lap_time:5.2f}s"
    val = (
        ""
        if np.isnan(rec.val_progress_pct)
        else f" | NOVAS: {rec.val_progress_pct:5.1f}% volta, {rec.val_finished_pct:5.1f}% pistas"
    )
    return (
        f"ger {rec.generation:3d} | fit {rec.best_fitness:6.1f} (média {rec.mean_fitness:6.1f})"
        f" | melhor: {rec.best_progress_pct:5.1f}% volta, {rec.best_finished_pct:5.1f}% pistas,"
        f" volta {lap} | pop completou {rec.finished_pct:5.1f}%{val}"
        f" | div {rec.diversity:.3f} sigma {rec.sigma:.3f} | {rec.seconds:4.1f}s"
    )


def load_history(path: str | Path) -> list[GenerationRecord]:
    """Lê um ``history.csv`` de volta."""
    ints = {"generation", "injected"}
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    names = [f.name for f in fields(GenerationRecord)]
    return [
        GenerationRecord(**{k: int(r[k]) if k in ints else float(r[k]) for k in names})  # type: ignore[arg-type]
        for r in rows
    ]
