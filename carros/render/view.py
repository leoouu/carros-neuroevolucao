"""Renderização Pygame: pista, carros, raios e HUD.

A simulação nunca importa este módulo; é o render que lê o estado do ``World``.
Assim o treino roda headless (sem janela) e o mesmo mundo pode ser desenhado
no replay.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pygame

from carros.sim.car import car_corners
from carros.sim.geometry import FloatArray
from carros.sim.sensors import ray_directions
from carros.sim.track import Track
from carros.sim.world import Status, World

Color = tuple[int, int, int]

BACKGROUND: Color = (14, 17, 23)
ASPHALT: Color = (36, 41, 51)
BORDER: Color = (120, 130, 150)
CHECKPOINT: Color = (52, 60, 74)
NEXT_CHECKPOINT: Color = (90, 200, 140)
START_LINE: Color = (230, 230, 230)
RAY: Color = (255, 200, 80)
CAR: Color = (80, 170, 255)
CAR_DEAD: Color = (90, 90, 100)
TEXT: Color = (220, 225, 235)

STATUS_LABEL = {
    Status.ALIVE: "rodando",
    Status.CRASHED: "bateu",
    Status.STALLED: "parado demais",
    Status.TIMEOUT: "sem checkpoint",
    Status.FINISHED: "completou",
}


@dataclass(frozen=True)
class Camera:
    """Transformação mundo -> tela que encaixa a pista na janela."""

    scale: float
    offset: FloatArray

    @classmethod
    def fit(cls, track: Track, size: tuple[int, int], margin: int = 30) -> Camera:
        """Escala uniforme que faz a pista caber na janela, centralizada."""
        xmin, ymin, xmax, ymax = track.bounds()
        w, h = size[0] - 2 * margin, size[1] - 2 * margin
        scale = min(w / (xmax - xmin), h / (ymax - ymin))
        used = np.array([(xmax - xmin) * scale, (ymax - ymin) * scale])
        offset = np.array([margin, margin]) + (np.array([w, h]) - used) / 2
        return cls(scale=scale, offset=offset - np.array([xmin, ymin]) * scale)

    def to_screen(self, points: FloatArray) -> list[tuple[float, float]]:
        """Converte ``(P, 2)`` pontos do mundo em coordenadas de tela."""
        pts = np.asarray(points) * self.scale + self.offset
        return [(float(x), float(y)) for x, y in pts]


class Renderer:
    """Desenha um ``World`` numa superfície Pygame."""

    def __init__(self, track: Track, size: tuple[int, int]) -> None:
        self.track = track
        self.size = size
        self.camera = Camera.fit(track, size)
        self.font = pygame.font.SysFont("consolas,dejavusansmono,monospace", 18)
        self.track_surface = self._draw_track()

    def _draw_track(self) -> pygame.Surface:
        """Pré-desenha a pista (estática) uma única vez."""
        surf = pygame.Surface(self.size)
        surf.fill(BACKGROUND)
        cam = self.camera
        pygame.draw.polygon(surf, ASPHALT, cam.to_screen(self.track.outer))
        pygame.draw.polygon(surf, BACKGROUND, cam.to_screen(self.track.inner))
        for cp in self.track.checkpoints[1:]:
            self._segment(surf, CHECKPOINT, cp, 1)
        self._segment(surf, START_LINE, self.track.checkpoints[0], 3)
        pygame.draw.aalines(surf, BORDER, True, cam.to_screen(self.track.outer))
        pygame.draw.aalines(surf, BORDER, True, cam.to_screen(self.track.inner))
        return surf

    def draw(
        self,
        screen: pygame.Surface,
        world: World,
        show_sensors: bool = True,
        colors: list[Color] | None = None,
        trails: list[list[tuple[float, float]]] | None = None,
    ) -> None:
        """Desenha pista, próximos checkpoints, rastros, raios e carros.

        Args:
            colors: uma cor por carro (padrão: todos azuis).
            trails: posições passadas de cada carro, em coordenadas do mundo.
        """
        screen.blit(self.track_surface, (0, 0))
        cam = self.camera
        alive = world.alive
        for k in np.unique(world.next_checkpoint[alive]):
            self._segment(screen, NEXT_CHECKPOINT, self.track.checkpoints[k], 2)
        if trails:
            for i, trail in enumerate(trails):
                if len(trail) > 1:
                    color = colors[i] if colors else CAR
                    pygame.draw.aalines(screen, color, False, cam.to_screen(np.array(trail)))
        if show_sensors:
            self._draw_sensors(screen, world)
        corners = car_corners(world.state, world.cfg.car)
        out = (world.status != Status.ALIVE) & (world.status != Status.FINISHED)
        for i in np.argsort(~out):  # eliminados por baixo
            color = CAR_DEAD if out[i] else (colors[i] if colors else CAR)
            pygame.draw.polygon(screen, color, cam.to_screen(corners[i]))

    def _segment(self, surf: pygame.Surface, color: Color, seg: FloatArray, width: int) -> None:
        """Desenha um segmento ``[x1, y1, x2, y2]`` do mundo."""
        a, b = self.camera.to_screen(seg.reshape(2, 2))
        pygame.draw.line(surf, color, a, b, width)

    def _draw_sensors(self, screen: pygame.Surface, world: World) -> None:
        cam = self.camera
        params = world.cfg.sensors
        readings = world.sensors() * params.max_range
        dirs = ray_directions(world.state, params)
        origins = world.state.positions
        for i in np.flatnonzero(world.alive):
            hits = origins[i] + dirs[i] * readings[i][:, None]
            o = cam.to_screen(origins[i : i + 1])[0]
            for hx, hy in cam.to_screen(hits):
                pygame.draw.aaline(screen, RAY, o, (hx, hy))
                pygame.draw.circle(screen, RAY, (int(hx), int(hy)), 3)

    def draw_hud(self, screen: pygame.Surface, lines: list[str]) -> None:
        """Texto no canto superior esquerdo, sobre um painel translúcido."""
        rendered = [self.font.render(text, True, TEXT) for text in lines]
        width = max(s.get_width() for s in rendered) + 20
        height = sum(s.get_height() + 4 for s in rendered) + 12
        panel = pygame.Surface((width, height), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 150))
        screen.blit(panel, (10, 10))
        y = 16
        for surf in rendered:
            screen.blit(surf, (20, y))
            y += surf.get_height() + 4
