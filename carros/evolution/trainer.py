"""Laço de treino: avalia, registra, salva o melhor e gera a próxima população."""

from __future__ import annotations

import csv
import shutil
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from carros.brain.mlp import MLPSpec, random_genomes, save_genome
from carros.evolution.config import TrainConfig
from carros.evolution.evaluate import evaluate
from carros.evolution.ga import mutation_sigma, next_generation
from carros.evolution.operators import diversity
from carros.sim.config import SimConfig, load_sim_config
from carros.sim.track import load_track


@dataclass(frozen=True)
class GenerationRecord:
    """Uma linha do ``history.csv``."""

    generation: int
    best_fitness: float
    mean_fitness: float
    median_fitness: float
    best_progress: float
    finished_pct: float
    best_lap_time: float
    diversity: float
    sigma: float
    injected: int
    seconds: float


def network_spec(cfg: TrainConfig, sim_cfg: SimConfig) -> MLPSpec:
    """Entradas = sensores + velocidade; saídas = volante e acelerador."""
    return MLPSpec((sim_cfg.sensors.count + 1, *cfg.network.hidden, 2))


def _save_run_config(out: Path, cfg: TrainConfig, raw: dict[str, Any]) -> None:
    """Guarda a config do experimento e cópias da pista e da física usadas."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    shutil.copy(cfg.track, out / "track.yaml")
    shutil.copy(cfg.sim, out / "sim.yaml")


def train(
    cfg: TrainConfig, raw: dict[str, Any], log: Callable[[str], None] = print
) -> list[GenerationRecord]:
    """Executa o experimento inteiro e devolve o histórico por geração."""
    rng = np.random.default_rng(cfg.seed)
    track = load_track(cfg.track)
    sim_cfg = load_sim_config(cfg.sim)
    spec = network_spec(cfg, sim_cfg)
    out = Path(cfg.output_dir)
    _save_run_config(out, cfg, raw)
    log(f"Rede {spec.layer_sizes} -> {spec.n_params} genes; população {cfg.ga.population}")

    genomes = random_genomes(spec, cfg.ga.population, rng, cfg.network.init_scale)
    history: list[GenerationRecord] = []
    sigma, injected = 0.0, 0
    for gen in range(1, cfg.generations + 1):
        t0 = time.perf_counter()
        res = evaluate(genomes, spec, track, sim_cfg, cfg.episode, cfg.fitness)
        best = int(np.argmax(res.fitness))
        lap_times = res.lap_time[res.finished]
        rec = GenerationRecord(
            generation=gen,
            best_fitness=float(res.fitness[best]),
            mean_fitness=float(res.fitness.mean()),
            median_fitness=float(np.median(res.fitness)),
            best_progress=float(np.max(res.progress)),
            finished_pct=100.0 * float(res.finished.mean()),
            best_lap_time=float(np.min(lap_times)) if lap_times.size else float("nan"),
            diversity=diversity(genomes),
            sigma=sigma if gen > 1 else mutation_sigma(cfg.ga, 0),
            injected=injected,
            seconds=time.perf_counter() - t0,
        )
        history.append(rec)
        save_genome(
            out / f"best_gen{gen:03d}",
            genomes[best],
            spec,
            {
                "generation": gen,
                "fitness": rec.best_fitness,
                "progress": float(res.progress[best]),
                "finished": bool(res.finished[best]),
                "lap_time": None if np.isnan(res.lap_time[best]) else float(res.lap_time[best]),
                "experiment": cfg.name,
            },
        )
        _append_csv(out / "history.csv", rec, header=gen == 1)
        log(_format(rec, track.n_checkpoints))

        if gen < cfg.generations:
            off = next_generation(
                genomes, res.fitness, cfg.ga, gen - 1, cfg.network.init_scale, rng
            )
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


def _format(rec: GenerationRecord, n_cps: int) -> str:
    lap = "-" if np.isnan(rec.best_lap_time) else f"{rec.best_lap_time:5.2f}s"
    return (
        f"ger {rec.generation:3d} | melhor {rec.best_fitness:7.2f} | média {rec.mean_fitness:6.2f}"
        f" | progresso {rec.best_progress / n_cps:6.1%} da volta | completaram"
        f" {rec.finished_pct:5.1f}% | volta {lap} | div {rec.diversity:.3f}"
        f" | sigma {rec.sigma:.3f} | {rec.seconds:4.1f}s"
    )


def load_history(path: str | Path) -> list[GenerationRecord]:
    """Lê um ``history.csv`` de volta."""
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [
        GenerationRecord(
            generation=int(r["generation"]),
            injected=int(r["injected"]),
            **{k: float(v) for k, v in r.items() if k not in ("generation", "injected")},
        )
        for r in rows
    ]
