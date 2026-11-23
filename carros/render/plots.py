"""Gráficos de evolução (Matplotlib), legíveis em tela de celular."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sem janela: só gera o arquivo

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

from carros.evolution.trainer import GenerationRecord  # noqa: E402

BG = "#0e1117"
FG = "#dce1eb"
GRID = "#2a2f3a"
BEST = "#50aaff"
MEAN = "#ffc850"
DONE = "#5ac88c"
VAL = "#ff6e6e"


def _style(ax: Axes) -> None:
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG, labelsize=13)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(color=GRID, lw=0.8)


def _legend(ax: Axes) -> None:
    ax.legend(loc="lower right", fontsize=12, facecolor=BG, edgecolor=GRID, labelcolor=FG)


def plot_history(history: list[GenerationRecord], path: str | Path, title: str = "") -> None:
    """Painel de cima: fitness. Painel de baixo: quanto os carros completam.

    Se houve validação, o painel de baixo compara o melhor carro em pistas de treino
    com o mesmo carro em pistas que ele nunca viu (generalização).
    """
    gens = np.array([r.generation for r in history])
    has_val = any(not np.isnan(r.val_finished_pct) for r in history)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), dpi=150, sharex=True)
    fig.patch.set_facecolor(BG)
    for ax in (ax1, ax2):
        _style(ax)

    ax1.plot(gens, [r.best_fitness for r in history], color=BEST, lw=3, label="melhor")
    ax1.plot(gens, [r.mean_fitness for r in history], color=MEAN, lw=2.5, label="média")
    ax1.set_ylabel("fitness", color=FG, fontsize=15)
    _legend(ax1)

    pop = [r.finished_pct for r in history]
    ax2.fill_between(gens, pop, color=DONE, alpha=0.18, lw=0)
    ax2.plot(gens, pop, color=DONE, lw=2, label="população: % de voltas completas")
    if has_val:
        ax2.plot(
            gens,
            [r.best_finished_pct for r in history],
            color=BEST,
            lw=2.5,
            label="melhor carro: % das pistas de treino",
        )
        val = [
            (r.generation, r.val_finished_pct) for r in history if not np.isnan(r.val_finished_pct)
        ]
        vx, vy = zip(*val, strict=True)
        ax2.plot(
            vx,
            vy,
            color=VAL,
            lw=2.5,
            ls="--",
            marker="o",
            ms=6,
            label="melhor carro: % das pistas NUNCA vistas",
        )
    ax2.set_ylim(-3, 103)
    ax2.set_ylabel("% completou", color=FG, fontsize=15)
    ax2.set_xlabel("geração", color=FG, fontsize=15)
    _legend(ax2)

    if title:
        ax1.set_title(title, color=FG, fontsize=17, pad=12)
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
