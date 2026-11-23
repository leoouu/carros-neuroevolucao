"""Compara genomas salvos com os baselines (heurística e carro aleatório) em várias pistas.

Exemplos:
    # melhor genoma final em pistas nunca vistas
    python scripts/compare.py --checkpoint checkpoints/phase3_generalization/best_gen100.npy \\
        --tracks configs/tracks/unseen configs/tracks/chicane.yaml
    # várias gerações de um treino, com o histórico
    python scripts/compare.py --run checkpoints/phase3_generalization --generations 1,10,50,100 \\
        --tracks configs/tracks/unseen --out comparacao.md
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from carros.brain.mlp import MLPSpec, load_genome
from carros.evolution.config import EpisodeParams, FitnessParams, expand_track_paths
from carros.evolution.evaluate import (
    Controller,
    EvalResult,
    heuristic_controller,
    network_controller,
    random_controller,
    run_episode,
)
from carros.evolution.trainer import load_history
from carros.sim.config import SimConfig, load_sim_config
from carros.sim.track import Track, load_track

FITNESS = FitnessParams(time_penalty=0.02, lap_bonus=50.0)


@dataclass
class Agent:
    """Algo que dirige: um nome, quantos carros correm e o controlador."""

    name: str
    n_cars: int
    make: Callable[[], Controller]  # recriado por pista: o aleatório fica reprodutível


def parse_args() -> argparse.Namespace:
    """Argumentos de linha de comando."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--checkpoint", nargs="*", type=Path, default=[])
    p.add_argument("--run", type=Path, help="pasta de um treino (usa --generations)")
    p.add_argument("--generations", default="", help='ex.: "1,10,50,100"')
    p.add_argument("--tracks", nargs="+", required=True, help="arquivos .yaml ou pastas")
    p.add_argument("--sim", help="padrão: sim.yaml do treino ou configs/sim_default.yaml")
    p.add_argument("--max-time", type=float, default=40.0)
    p.add_argument("--random-cars", type=int, default=20)
    p.add_argument("--out", type=Path, help="salva o relatório em Markdown")
    return p.parse_args()


def build_agents(args: argparse.Namespace, sim_cfg: SimConfig) -> list[Agent]:
    """Genomas pedidos + heurística + aleatório."""
    paths = list(args.checkpoint)
    if args.run and args.generations:
        paths += [args.run / f"best_gen{int(g):03d}.npy" for g in args.generations.split(",")]
    agents = []
    for path in paths:
        genome, spec, meta = load_genome(path)
        name = f"geração {meta['generation']}" if "generation" in meta else path.stem

        def make(genome: np.ndarray = genome, spec: MLPSpec = spec) -> Controller:
            return network_controller(spec, genome[None])

        agents.append(Agent(name, 1, make))
    agents.append(Agent("heurística", 1, lambda: heuristic_controller(sim_cfg)))
    agents.append(
        Agent(
            f"aleatório (média de {args.random_cars})",
            args.random_cars,
            lambda: random_controller(seed=0),
        )
    )
    return agents


def main() -> None:
    """Roda todos os agentes em todas as pistas e imprime as tabelas."""
    args = parse_args()
    default_sim = args.run / "sim.yaml" if args.run else Path("configs/sim_default.yaml")
    sim_cfg = load_sim_config(args.sim or default_sim)
    tracks: list[Track] = [load_track(p) for p in expand_track_paths(args.tracks)]
    episode = EpisodeParams(max_time=args.max_time, target_laps=1)
    agents = build_agents(args, sim_cfg)

    results: dict[str, list[EvalResult]] = {a.name: [] for a in agents}
    for track in tracks:
        for agent in agents:
            controller = agent.make()
            res = run_episode(track, sim_cfg, agent.n_cars, controller, episode, FITNESS)
            results[agent.name].append(res)

    lines = _summary_table(results, len(tracks)) + [""] + _per_track_table(results, tracks)
    if args.run and (args.run / "history.csv").exists():
        lines += [""] + _history_table(args.run / "history.csv", args.generations)
    report = "\n".join(lines)
    print(report)
    if args.out:
        args.out.write_text(report + "\n", encoding="utf-8")


def _summary_table(results: dict[str, list[EvalResult]], n_tracks: int) -> list[str]:
    rows = [
        f"## Resumo em {n_tracks} pista(s)",
        "",
        "| agente | % média da volta | pistas completadas | tempo médio de volta* |",
        "|---|---:|---:|---:|",
    ]
    for name, res in results.items():
        progress = 100 * np.mean([r.progress.mean() for r in res])
        done = np.mean([r.finished.mean() for r in res])
        laps = [r.lap_time[r.finished].mean() for r in res if r.finished.any()]
        lap = f"{np.mean(laps):.2f} s" if laps else "—"
        rows.append(f"| {name} | {progress:.1f}% | {100 * done:.0f}% | {lap} |")
    rows += ["", "\\* média só nas pistas que o agente completou."]
    return rows


def _per_track_table(results: dict[str, list[EvalResult]], tracks: list[Track]) -> list[str]:
    names = list(results)
    rows = [
        "## % da volta por pista (✓ = completou, com o tempo)",
        "",
        "| pista | " + " | ".join(names) + " |",
        "|---|" + "---:|" * len(names),
    ]
    for i, track in enumerate(tracks):
        cells = []
        for name in names:
            r = results[name][i]
            if r.finished.all():
                cells.append(f"✓ {r.lap_time.mean():.2f} s")
            else:
                cells.append(f"{100 * r.progress.mean():.0f}%")
        rows.append(f"| {track.name} | " + " | ".join(cells) + " |")
    return rows


def _history_table(path: Path, generations: str) -> list[str]:
    hist = {r.generation: r for r in load_history(path)}
    wanted = [int(g) for g in generations.split(",")] if generations else sorted(hist)
    rows = [
        "## Histórico do treino (pistas de treino)",
        "",
        "| geração | fitness máx | fitness médio | % da população que completa |"
        " melhor: % pistas | melhor: tempo médio | NOVAS: % pistas |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for g in wanted:
        r = hist[g]
        lap = "—" if np.isnan(r.best_lap_time) else f"{r.best_lap_time:.2f} s"
        val = "—" if np.isnan(r.val_finished_pct) else f"{r.val_finished_pct:.0f}%"
        rows.append(
            f"| {g} | {r.best_fitness:.1f} | {r.mean_fitness:.1f} | {r.finished_pct:.1f}% |"
            f" {r.best_finished_pct:.0f}% | {lap} | {val} |"
        )
    return rows


if __name__ == "__main__":
    main()
