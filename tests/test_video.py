"""Exportação de vídeo (headless)."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import imageio_ffmpeg  # noqa: E402
import numpy as np  # noqa: E402
import pygame  # noqa: E402

from carros.evolution.config import EpisodeParams, FitnessParams  # noqa: E402
from carros.evolution.evaluate import heuristic_controller, run_episode  # noqa: E402
from carros.render.video import HEADER, Panel, grid_frames, record, write_mp4  # noqa: E402
from carros.sim.config import SimConfig  # noqa: E402
from carros.sim.track import Track  # noqa: E402

FIT = FitnessParams(time_penalty=0.02, lap_bonus=50.0)


def test_recording_matches_evaluation(cfg: SimConfig, simple_track: Track) -> None:
    """O vídeo mostra exatamente o episódio que a avaliação mede."""
    ep = EpisodeParams(30.0, 1)
    rec = record(simple_track, cfg, 2, heuristic_controller(cfg), ep, FIT)
    res = run_episode(simple_track, cfg, 2, heuristic_controller(cfg), ep, FIT)
    np.testing.assert_array_equal(rec.status[-1], res.status)
    np.testing.assert_allclose(rec.best_fitness[-1], res.fitness.max())


def test_mp4_has_expected_size_and_duration(
    tmp_path: Path, cfg: SimConfig, simple_track: Track
) -> None:
    pygame.init()
    pygame.display.set_mode((1, 1))
    ep = EpisodeParams(0.5, 1)
    rec = record(simple_track, cfg, 1, heuristic_controller(cfg), ep, FIT)
    size = (320, HEADER + 200)
    frames = grid_frames([Panel(rec, (320, 200), "x", head=40)], 1, size, "t", "s", 30)
    seconds = write_mp4(tmp_path / "v.mp4", frames, 30)
    meta = next(imageio_ffmpeg.read_frames(str(tmp_path / "v.mp4")))
    assert meta["size"] == size
    assert abs(meta["duration"] - seconds) < 0.1
    assert meta["codec"] == "h264" and meta["pix_fmt"].startswith("yuv420p")
