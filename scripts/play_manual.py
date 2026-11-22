"""Dirija o carro pelo teclado para validar física, pista, sensores e colisão.

Controles: setas ou WASD | R reinicia | TAB mostra/esconde sensores
           H liga/desliga o piloto automático (heurística) | ESC sai

Exemplo:
    python scripts/play_manual.py --track configs/tracks/simple.yaml
    python scripts/play_manual.py --autopilot --screenshot frame.png --frames 240
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pygame

from carros.baselines import heuristic_policy
from carros.render.view import STATUS_LABEL, Renderer
from carros.sim.config import load_sim_config
from carros.sim.sensors import ray_offsets
from carros.sim.track import load_track
from carros.sim.world import Status, World

WINDOW = (1100, 760)


def parse_args() -> argparse.Namespace:
    """Argumentos de linha de comando."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--track", default="configs/tracks/simple.yaml")
    p.add_argument("--sim", default="configs/sim_default.yaml")
    p.add_argument(
        "--strict",
        action="store_true",
        help="aplica também eliminação por ficar parado / sem checkpoint",
    )
    p.add_argument("--autopilot", action="store_true", help="começa com a heurística dirigindo")
    p.add_argument("--screenshot", type=Path, help="salva um PNG após --frames quadros e sai")
    p.add_argument("--frames", type=int, default=180)
    return p.parse_args()


def keyboard_controls() -> tuple[float, float]:
    """Lê o teclado e devolve ``(steer, throttle)`` em ``[-1, 1]``."""
    keys = pygame.key.get_pressed()
    steer = float(keys[pygame.K_RIGHT] or keys[pygame.K_d]) - float(
        keys[pygame.K_LEFT] or keys[pygame.K_a]
    )
    throttle = float(keys[pygame.K_UP] or keys[pygame.K_w]) - float(
        keys[pygame.K_DOWN] or keys[pygame.K_s]
    )
    return steer, throttle


def hud_lines(world: World, autopilot: bool) -> list[str]:
    """Texto do painel de informações."""
    best = world.first_lap_time[0]
    lap = "-" if np.isnan(best) else f"{best:5.2f}s"
    return [
        f"modo: {'piloto automático' if autopilot else 'manual'}",
        f"velocidade: {world.state.speed[0]:6.1f} px/s",
        f"checkpoint: {world.checkpoints_passed[0] % world.track.n_checkpoints}"
        f"/{world.track.n_checkpoints}   voltas: {world.laps[0]}",
        f"tempo: {world.time:6.2f}s   1ª volta: {lap}",
        f"status: {STATUS_LABEL[Status(world.status[0])]}",
        "setas/WASD  R reinicia  TAB sensores  H piloto",
    ]


def main() -> None:
    """Loop principal: física em passo fixo, render a 60 fps."""
    args = parse_args()
    track = load_track(args.track)
    cfg = load_sim_config(args.sim)
    world = World(track, cfg, n_cars=1, eliminate_idle=args.strict)

    pygame.init()
    screen = pygame.display.set_mode(WINDOW)
    pygame.display.set_caption(f"Carros - pista '{track.name}'")
    renderer = Renderer(track, WINDOW)
    clock = pygame.time.Clock()
    offsets = ray_offsets(cfg.sensors)
    show_sensors, autopilot, frame = True, args.autopilot, 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                world.reset()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_h:
                autopilot = not autopilot
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_TAB:
                show_sensors = not show_sensors

        if world.any_alive():
            if autopilot:
                steer_arr, throttle_arr = heuristic_policy(world.sensors(), offsets)
            else:
                steer, throttle = keyboard_controls()
                steer_arr, throttle_arr = np.array([steer]), np.array([throttle])
            world.step(steer_arr, throttle_arr)

        renderer.draw(screen, world, show_sensors)
        renderer.draw_hud(screen, hud_lines(world, autopilot))
        pygame.display.flip()
        clock.tick(round(1.0 / cfg.dt))

        frame += 1
        if args.screenshot and frame >= args.frames:
            pygame.image.save(screen, str(args.screenshot))
            running = False
    pygame.quit()


if __name__ == "__main__":
    main()
