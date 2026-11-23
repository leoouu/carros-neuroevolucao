"""Gera os vídeos do projeto (MP4 para o LinkedIn) sem abrir janela.

Exemplos:
    # gerações lado a lado (1920x1080), a população inteira
    python scripts/make_video.py side --run checkpoints/X --generations 1,10,100 \\
        --track configs/tracks/test/test_2000.yaml --out videos/lado_a_lado.mp4
    # gerações em sequência (1080x1350, bom para celular)
    python scripts/make_video.py story --run checkpoints/X --generations 1,5,10,100 ...
    # um carro em várias pistas nunca vistas (grade 3x3, 1080x1350)
    python scripts/make_video.py tracks --run checkpoints/X --generations 100 \\
        --track configs/tracks/test --out videos/pistas_novas.mp4
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from collections.abc import Iterator
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np  # noqa: E402
import pygame  # noqa: E402

from carros.brain.mlp import MLPSpec, load_genome  # noqa: E402
from carros.evolution.config import (  # noqa: E402
    TrainConfig,
    expand_track_paths,
    load_train_config,
)
from carros.evolution.evaluate import network_controller  # noqa: E402
from carros.evolution.trainer import population_path  # noqa: E402
from carros.render.video import (  # noqa: E402
    HEADER,
    Panel,
    Recording,
    grid_frames,
    record,
    sequence_frames,
    write_mp4,
)
from carros.sim.config import load_sim_config  # noqa: E402
from carros.sim.geometry import FloatArray  # noqa: E402
from carros.sim.track import load_track  # noqa: E402

FPS = 30


def parse_args() -> argparse.Namespace:
    """Argumentos de linha de comando."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("mode", choices=["side", "story", "tracks"])
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--generations", required=True)
    p.add_argument("--track", nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--title", default="Carros que aprendem a dirigir")
    p.add_argument("--subtitle", default="Neuroevolução: ninguém programou como dirigir")
    p.add_argument("--speed", type=float, default=2.0, help="aceleração no modo story")
    p.add_argument("--max-time", type=float, default=12.0, help="corte da simulação (s)")
    return p.parse_args()


def main() -> None:
    """Simula, desenha e grava."""
    args = parse_args()
    pygame.init()
    pygame.display.set_mode((1, 1))
    cfg, _ = load_train_config(args.run / "config.yaml")
    gens = [int(g) for g in args.generations.split(",")]
    tracks = expand_track_paths(args.track)
    frames: Iterator[FloatArray]
    if args.mode == "tracks":
        recs = [_record_best(args, cfg, gens[-1], t) for t in tracks[:9]]
        panels = [Panel(r, (360, 400), f"Pista {i + 1}", False, 58) for i, r in enumerate(recs)]
        frames = grid_frames(panels, 3, (1080, 1350), args.title, args.subtitle, FPS)
    else:
        recs = [_record_population(args, cfg, g, tracks[0]) for g in gens]
        size = (1920 // len(gens), 1080 - HEADER) if args.mode == "side" else (1080, 1350 - HEADER)
        head = 92 if args.mode == "side" else 110
        panels = [
            Panel(r, size, f"Geração {g}", head=head) for r, g in zip(recs, gens, strict=True)
        ]
        if args.mode == "side":
            frames = grid_frames(panels, len(gens), (1920, 1080), args.title, args.subtitle, FPS)
        else:
            frames = sequence_frames(
                panels, (1080, 1350), args.title, args.subtitle, FPS, args.speed
            )
    print(f"{args.out}: {write_mp4(args.out, frames, FPS):.1f} s")


def _record_population(
    args: argparse.Namespace, cfg: TrainConfig, gen: int, track: str
) -> Recording:
    _, spec, _ = load_genome(args.run / f"best_gen{gen:03d}")
    return _record(args, cfg, spec, np.load(population_path(args.run, gen)), track)


def _record_best(args: argparse.Namespace, cfg: TrainConfig, gen: int, track: str) -> Recording:
    genome, spec, _ = load_genome(args.run / f"best_gen{gen:03d}")
    return _record(args, cfg, spec, genome[None], track)


def _record(
    args: argparse.Namespace, cfg: TrainConfig, spec: MLPSpec, genomes: FloatArray, track: str
) -> Recording:
    sim = load_sim_config(args.run / "sim.yaml")
    episode = dataclasses.replace(cfg.episode, max_time=args.max_time)
    controller = network_controller(spec, genomes)
    return record(load_track(track), sim, len(genomes), controller, episode, cfg.fitness)


if __name__ == "__main__":
    main()
