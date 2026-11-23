"""Gráficos de evolução (Matplotlib), legíveis em tela de celular."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sem janela: só gera o arquivo

import matplotlib.pyplot as plt  # noqa: E402

from carros.evolution.trainer import GenerationRecord  # noqa: E402

BG = "#0e1117"
FG = "#dce1eb"
GRID = "#2a2f3a"
BEST = "#50aaff"
MEAN = "#ffc850"
DONE = "#5ac88c"


def plot_history(history: list[GenerationRecord], path: str | Path, title: str = "") -> None:
    """Fitness máximo/médio e % de carros que completaram a volta, por geração."""
    gens = [r.generation for r in history]
    fig, ax = plt.subplots(figsize=(9, 5.4), dpi=150)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.plot(gens, [r.best_fitness for r in history], color=BEST, lw=3, label="melhor fitness")
    ax.plot(gens, [r.mean_fitness for r in history], color=MEAN, lw=2.5, label="fitness médio")
    ax.set_xlabel("geração", color=FG, fontsize=15)
    ax.set_ylabel("fitness", color=FG, fontsize=15)

    ax2 = ax.twinx()
    ax2.set_facecolor(BG)
    ax2.fill_between(gens, [r.finished_pct for r in history], color=DONE, alpha=0.18, lw=0)
    ax2.plot(gens, [r.finished_pct for r in history], color=DONE, lw=2, label="% completou a volta")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("% completou a volta", color=DONE, fontsize=15)

    for a in (ax, ax2):
        a.tick_params(colors=FG, labelsize=13)
        for spine in a.spines.values():
            spine.set_color(GRID)
    ax.grid(color=GRID, lw=0.8)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(
        lines,
        [str(ln.get_label()) for ln in lines],
        loc="upper left",
        fontsize=13,
        facecolor=BG,
        edgecolor=GRID,
        labelcolor=FG,
    )
    if title:
        ax.set_title(title, color=FG, fontsize=17, pad=12)
    fig.tight_layout()
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
