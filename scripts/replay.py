"""Reproduz um ou mais genomas salvos, juntos na mesma pista, com rastro.

Exemplos:
    python scripts/replay.py --checkpoint checkpoints/phase2_simple/best_gen050.npy
    python scripts/replay.py --checkpoint checkpoints/phase2_simple/best_gen001.npy \\
        checkpoints/phase2_simple/best_gen002.npy checkpoints/phase2_simple/best_gen050.npy
    python scripts/replay.py --checkpoint ... --screenshot replay.png --frames 300

Controles: R reinicia | TAB sensores | ESC sai. (Para exportar MP4, use make_video.py.)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pygame

from carros.brain.mlp import MLPSpec, forward, load_genome
from carros.render.view import STATUS_LABEL, Color, Renderer
from carros.sim.config import load_sim_config
from carros.sim.track import load_track
from carros.sim.world import Status, World

WINDOW = (1100, 760)
PALETTE: list[Color] = [(255, 110, 110), (255, 200, 80), (80, 170, 255), (90, 220, 150)]


def parse_args() -> argparse.Namespace:
    """Argumentos de linha de comando."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--checkpoint", nargs="+", type=Path, required=True)
    p.add_argument("--track", help="padrão: 1ª pista de treino salva junto do checkpoint")
    p.add_argument("--sim", help="padrão: sim.yaml salvo junto do checkpoint")
    p.add_argument("--max-time", type=float, default=40.0)
    p.add_argument("--screenshot", type=Path, help="salva um PNG após --frames quadros e sai")
    p.add_argument("--frames", type=int, default=300)
    return p.parse_args()


def main() -> None:
    """Carrega os genomas e roda o replay em tempo real."""
    args = parse_args()
    loaded = [load_genome(path) for path in args.checkpoint]
    run_dir = args.checkpoint[0].parent
    track = load_track(args.track or sorted((run_dir / "tracks").glob("*.yaml"))[0])
    cfg = load_sim_config(args.sim or run_dir / "sim.yaml")
    world = World(track, cfg, n_cars=len(loaded), target_laps=1)
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(loaded))]
    labels = [f"geração {meta.get('generation', '?')}" for _, _, meta in loaded]
    trails: list[list[tuple[float, float]]] = [[] for _ in loaded]

    pygame.init()
    screen = pygame.display.set_mode(WINDOW)
    pygame.display.set_caption(f"Replay - pista '{track.name}'")
    renderer = Renderer(track, WINDOW)
    clock = pygame.time.Clock()
    show_sensors, frame, running = len(loaded) == 1, 0, True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                world.reset()
                trails = [[] for _ in loaded]
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_TAB:
                show_sensors = not show_sensors

        if world.any_alive() and world.time < args.max_time:
            _step(world, loaded)
            for i, (x, y) in enumerate(world.state.positions):
                if world.alive[i]:
                    trails[i].append((float(x), float(y)))

        renderer.draw(screen, world, show_sensors, colors=colors, trails=trails)
        renderer.draw_hud(screen, _hud(world, labels))
        _legend(screen, renderer, labels, colors)
        pygame.display.flip()
        clock.tick(round(1.0 / cfg.dt))

        frame += 1
        if args.screenshot and frame >= args.frames:
            pygame.image.save(screen, str(args.screenshot))
            running = False
    pygame.quit()


def _step(world: World, loaded: list[tuple[np.ndarray, MLPSpec, dict[str, object]]]) -> None:
    """Cada carro usa sua própria rede (arquiteturas podem diferir entre checkpoints)."""
    obs = world.observe()
    controls = np.zeros((world.n_cars, 2))
    for i, (genome, spec, _) in enumerate(loaded):
        if world.alive[i]:
            controls[i] = forward(spec, genome[None], obs[i : i + 1])[0]
    world.step(controls[:, 0], controls[:, 1])


def _hud(world: World, labels: list[str]) -> list[str]:
    lines = [f"tempo: {world.time:5.2f}s"]
    for i, label in enumerate(labels):
        lap = world.first_lap_time[i]
        extra = f" em {lap:.2f}s" if world.status[i] == Status.FINISHED else ""
        pct = world.progress()[i] / world.track.n_checkpoints
        lines.append(f"{label}: {pct:4.0%} {STATUS_LABEL[Status(world.status[i])]}{extra}")
    return lines


def _legend(
    screen: pygame.Surface, renderer: Renderer, labels: list[str], colors: list[Color]
) -> None:
    """Quadradinhos coloridos ao lado de cada linha do HUD."""
    y = 16 + renderer.font.get_linesize() + 4
    for color in colors[: len(labels)]:
        pygame.draw.rect(screen, color, (4, y + 4, 10, 10))
        y += renderer.font.get_linesize() + 4


if __name__ == "__main__":
    main()
