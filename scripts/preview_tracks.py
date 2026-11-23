"""Desenha várias pistas lado a lado num PNG (para conferir pistas novas).

Exemplo:
    python scripts/preview_tracks.py configs/tracks/*.yaml --out tracks.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402

from carros.render.view import BACKGROUND, TEXT, Renderer  # noqa: E402
from carros.sim.track import load_track  # noqa: E402

TILE = (440, 320)


def main() -> None:
    """Monta a grade de miniaturas e salva."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("tracks", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=Path("tracks.png"))
    p.add_argument("--cols", type=int, default=4)
    args = p.parse_args()

    paths = sorted(path for pattern in args.tracks for path in _expand(pattern))
    pygame.init()
    pygame.display.set_mode((1, 1))
    cols = min(args.cols, len(paths))
    rows = (len(paths) + cols - 1) // cols
    sheet = pygame.Surface((cols * TILE[0], rows * TILE[1]))
    sheet.fill(BACKGROUND)
    for i, path in enumerate(paths):
        track = load_track(path)
        tile = Renderer(track, TILE).track_surface.copy()
        label = f"{track.name}  ({track.n_checkpoints} cps, largura {track.width:.0f})"
        tile.blit(pygame.font.SysFont("consolas", 15).render(label, True, TEXT), (8, 6))
        sheet.blit(tile, ((i % cols) * TILE[0], (i // cols) * TILE[1]))
    pygame.image.save(sheet, str(args.out))
    print(f"{len(paths)} pistas -> {args.out}")


def _expand(path: Path) -> list[Path]:
    """Aceita arquivos ou pastas (pega os .yaml de dentro)."""
    return sorted(path.glob("*.yaml")) if path.is_dir() else [path]


if __name__ == "__main__":
    main()
