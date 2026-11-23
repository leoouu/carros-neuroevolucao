"""Exportação de vídeos: grava episódios headless e desenha painéis em MP4/GIF.

Primeiro simula e guarda a trajetória inteira; só depois desenha. Assim o vídeo
já sabe quem será o melhor carro (para destacá-lo) e pode abrir pelo quadro final.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pygame

from carros.evolution.config import EpisodeParams, FitnessParams
from carros.evolution.evaluate import Controller
from carros.evolution.fitness import compute_fitness
from carros.render.view import BACKGROUND, BORDER, TEXT, Color, Renderer
from carros.sim.car import CarState, car_corners
from carros.sim.config import CarParams, SimConfig
from carros.sim.geometry import FloatArray
from carros.sim.track import Track
from carros.sim.world import Status, World

BEST: Color = (255, 205, 60)
DEAD: Color = (120, 58, 66)
MUTED: Color = (190, 198, 212)
DONE: Color = (90, 220, 150)
COOL: Color = (70, 140, 255)
WARM: Color = (80, 225, 235)
SS = 2  # superamostragem: desenha em 2x e reduz, para bordas lisas


@dataclass(frozen=True)
class Recording:
    """Trajetória de todos os carros: arrays ``(T, N)``."""

    track: Track
    car: CarParams
    dt: float
    states: list[CarState]
    status: FloatArray
    best_fitness: FloatArray  # (T,) melhor fitness até o momento, nesta pista
    best: int

    def step_at(self, seconds: float) -> int:
        """Índice do passo no instante pedido (congela no fim)."""
        return min(int(round(seconds / self.dt)), len(self.states) - 1)

    @property
    def duration(self) -> float:
        """Duração em segundos."""
        return (len(self.states) - 1) * self.dt


def record(
    track: Track,
    sim: SimConfig,
    n_cars: int,
    controller: Controller,
    episode: EpisodeParams,
    fitness: FitnessParams,
) -> Recording:
    """Roda um episódio guardando o estado de cada passo."""
    world = World(track, sim, n_cars, target_laps=episode.target_laps)
    states, status, best = [world.state], [world.status.copy()], [0.0]
    scores = np.zeros(n_cars)
    for _ in range(int(round(episode.max_time / sim.dt))):
        if not world.any_alive():
            break
        world.step(*controller(world))
        finished = world.status == Status.FINISHED
        progress = world.progress() / track.n_checkpoints
        scores = compute_fitness(progress, finished, world.time_alive, fitness)
        states.append(world.state)
        status.append(world.status.copy())
        best.append(float(np.max(scores)))
    return Recording(
        track=track,
        car=sim.car,
        dt=sim.dt,
        states=states,
        status=np.array(status, dtype=np.float64),
        best_fitness=np.array(best),
        best=int(np.argmax(scores)),
    )


def font(size: int, bold: bool = False) -> pygame.font.Font:
    """Fonte sem serifa do sistema, já na escala de superamostragem."""
    return pygame.font.SysFont("segoeui,dejavusans,arial", size * SS, bold=bold)


def car_colors(n: int) -> list[Color]:
    """Degradê azul -> ciano, para os carros da população se distinguirem."""
    mix = np.linspace(0.0, 1.0, n)[:, None]
    rgb = (1 - mix) * np.array(COOL) + mix * np.array(WARM)
    return [(int(r), int(g), int(b)) for r, g, b in rgb]


class Panel:
    """Um episódio desenhado num retângulo do vídeo, com título e HUD."""

    def __init__(
        self, rec: Recording, size: tuple[int, int], title: str, stats: bool = True, head: int = 92
    ) -> None:
        self.rec, self.title, self.stats, self.head = rec, title, stats, head * SS
        self.size = (size[0] * SS, size[1] * SS)
        self.area = (self.size[0], self.size[1] - self.head)
        self.renderer = Renderer(rec.track, self.area)
        for border in (rec.track.outer, rec.track.inner):  # borda grossa: legível no celular
            points = self.renderer.camera.to_screen(border)
            pygame.draw.lines(self.renderer.track_surface, BORDER, True, points, 2 * SS)
        self.colors = car_colors(rec.status.shape[1])
        self.scale = self.renderer.camera.scale
        self.big, self.small = font(head * 36 // 92, True), font(head * 28 // 92, True)
        self.reset()

    def reset(self) -> None:
        """Apaga os rastros e volta ao início."""
        self.trails = pygame.Surface(self.area, pygame.SRCALPHA)
        self.last = 0

    def draw(self, target: pygame.Surface, pos: tuple[int, int], seconds: float) -> None:
        """Desenha o instante pedido em ``target`` (``pos`` em pixels finais)."""
        step = self.rec.step_at(seconds)
        self._extend_trails(step)
        view = self.renderer.track_surface.copy()
        view.blit(self.trails, (0, 0))
        self._draw_cars(view, step)
        x, y = pos[0] * SS, pos[1] * SS
        target.blit(view, (x, y + self.head))
        self._draw_head(target, x, y, step)

    def _extend_trails(self, step: int) -> None:
        cam = self.renderer.camera
        for k in range(self.last + 1, step + 1):
            if k % 4 == 0:  # rastro esmaece aos poucos (o do melhor é redesenhado inteiro)
                self.trails.fill((0, 0, 0, 2), special_flags=pygame.BLEND_RGBA_SUB)
            before = cam.to_screen(self.rec.states[k - 1].positions)
            after = cam.to_screen(self.rec.states[k].positions)
            for i in np.flatnonzero(self.rec.status[k - 1] == Status.ALIVE):
                pygame.draw.line(self.trails, (*self.colors[i], 95), before[i], after[i], SS)
        self.last = max(self.last, step)

    def _draw_cars(self, view: pygame.Surface, step: int) -> None:
        cam, best = self.renderer.camera, self.rec.best
        path = np.array([s.positions[best] for s in self.rec.states[: step + 1]])
        if len(path) > 1:
            pygame.draw.lines(view, BEST, False, cam.to_screen(path), 2 * SS)
        corners = car_corners(self.rec.states[step], self.rec.car)
        status = self.rec.status[step]
        out = (status != Status.ALIVE) & (status != Status.FINISHED)
        for i in np.argsort(~out, kind="stable"):  # eliminados por baixo
            if i != best:
                color = DEAD if out[i] else self.colors[i]
                pygame.draw.polygon(view, color, cam.to_screen(corners[i]))
        shape = cam.to_screen(corners[best])
        pygame.draw.polygon(view, DEAD if out[best] else BEST, shape)
        pygame.draw.polygon(view, BEST if out[best] else (255, 255, 255), shape, SS)

    def _draw_head(self, target: pygame.Surface, x: int, y: int, step: int) -> None:
        center = x + self.size[0] // 2
        name = self.big.render(self.title, True, TEXT)
        target.blit(name, (center - name.get_width() // 2, y + 2 * SS))
        status = self.rec.status[step]
        done = int(np.sum(status == Status.FINISHED))
        if self.stats:
            text = f"completaram {done}/{len(status)}  ·  fitness {self.rec.best_fitness[step]:.0f}"
            color = DONE if done else MUTED
        elif done:
            text, color = "completou", DONE
        else:
            alive = status[0] == Status.ALIVE
            text, color = ("rodando", MUTED) if alive else ("bateu", (235, 100, 100))
        info = self.small.render(text, True, color)
        target.blit(info, (center - info.get_width() // 2, y + int(self.head * 0.56)))


def _header(canvas: pygame.Surface, title: str, subtitle: str, width: int) -> None:
    for text, size, bold, color, y in (
        (title, 46, True, TEXT, 22),
        (subtitle, 28, False, MUTED, 82),
    ):
        surf = font(size, bold).render(text, True, color)
        canvas.blit(surf, ((width * SS - surf.get_width()) // 2, y * SS))


def _finish(canvas: pygame.Surface, size: tuple[int, int]) -> FloatArray:
    small = pygame.transform.smoothscale(canvas, size)
    return np.transpose(pygame.surfarray.array3d(small), (1, 0, 2))


HEADER = 128


def grid_frames(
    panels: list[Panel],
    cols: int,
    size: tuple[int, int],
    title: str,
    subtitle: str,
    fps: int,
    intro: float = 1.0,
    outro: float = 1.5,
) -> Iterator[FloatArray]:
    """Todos os painéis ao mesmo tempo. Abre com o quadro final (miniatura do post)."""
    end = max(p.rec.duration for p in panels)
    cell = (panels[0].size[0] // SS, panels[0].size[1] // SS)

    def frame(seconds: float) -> FloatArray:
        canvas = pygame.Surface((size[0] * SS, size[1] * SS))
        canvas.fill(BACKGROUND)
        _header(canvas, title, subtitle, size[0])
        for i, panel in enumerate(panels):
            pos = ((i % cols) * cell[0], HEADER + (i // cols) * cell[1])
            panel.draw(canvas, pos, seconds)
        return _finish(canvas, size)

    yield from [frame(end)] * int(intro * fps)
    for panel in panels:
        panel.reset()
    for k in range(int(end * fps) + 1):
        last = frame(k / fps)
        yield last
    yield from [frame(end)] * int(outro * fps)


def sequence_frames(
    panels: list[Panel],
    size: tuple[int, int],
    title: str,
    subtitle: str,
    fps: int,
    speed: float,
    intro: float = 1.0,
    pause: float = 0.6,
) -> Iterator[FloatArray]:
    """Um painel após o outro, acelerado ``speed`` vezes. Abre com o final do último."""

    def frame(panel: Panel, seconds: float) -> FloatArray:
        canvas = pygame.Surface((size[0] * SS, size[1] * SS))
        canvas.fill(BACKGROUND)
        _header(canvas, title, subtitle, size[0])
        panel.draw(canvas, (0, HEADER), seconds)
        return _finish(canvas, size)

    yield from [frame(panels[-1], panels[-1].rec.duration)] * int(intro * fps)
    panels[-1].reset()
    for panel in panels:
        for k in range(int(panel.rec.duration / speed * fps) + 1):
            yield frame(panel, k / fps * speed)
        yield from [frame(panel, panel.rec.duration)] * int(pause * fps)


def write_mp4(path: Path, frames: Iterable[FloatArray], fps: int) -> float:
    """Grava H.264 (yuv420p, o formato aceito pelo LinkedIn); devolve a duração em s."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count, writer = 0, None
    for frame in frames:
        if writer is None:
            height, width = frame.shape[:2]
            writer = imageio_ffmpeg.write_frames(
                str(path),
                (width, height),
                fps=fps,
                codec="libx264",
                quality=None,  # a qualidade vem do -crf abaixo
                macro_block_size=2,
                output_params=["-crf", "18", "-preset", "slow", "-movflags", "+faststart"],
            )
            writer.send(None)  # inicia o ffmpeg
        writer.send(np.ascontiguousarray(frame, dtype=np.uint8))
        count += 1
    if writer is not None:
        writer.close()
    return count / fps
