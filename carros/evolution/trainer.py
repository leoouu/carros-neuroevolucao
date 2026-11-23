"""Laço de treino: avalia, registra, salva o melhor e gera a próxima população."""

from __future__ import annotations

import csv
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


def train(
    cfg: TrainConfig, raw: dict[str, Any], log: Callable[[str], None] = print
) -> list[GenerationRecord]:
    """Executa o experimento inteiro e devolve o histórico por geração."""
    rng = np.random.default_rng(cfg.seed)
    out = Path(cfg.output_dir)
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
        sigma, injected = mutation_sigma(cfg.ga, 0), 0
        for gen in range(1, cfg.generations + 1):
            t0 = time.perf_counter()
            results = train_set.evaluate(genomes, spec, cfg.episode, cfg.fitness)
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
