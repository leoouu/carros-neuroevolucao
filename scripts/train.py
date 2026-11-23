"""Treina uma população de carros por neuroevolução.

Exemplo:
    python scripts/train.py --config configs/phase2_simple_track.yaml
    python scripts/train.py --config configs/phase3_generalization.yaml
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from carros.evolution.config import load_train_config
from carros.evolution.trainer import train
from carros.render.plots import plot_history


def main() -> None:
    """Lê a config, treina e salva o gráfico de fitness."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", default="configs/phase2_simple_track.yaml")
    p.add_argument("--generations", type=int, help="sobrescreve o número de gerações")
    args = p.parse_args()

    cfg, raw = load_train_config(args.config)
    if args.generations:
        raw["generations"] = args.generations  # a config salva reflete o que rodou
        cfg = dataclasses.replace(cfg, generations=args.generations)
    history = train(cfg, raw)
    out = Path(cfg.output_dir)
    plot_history(history, out / "fitness.png", title=f"Neuroevolução — {cfg.name}")
    print(f"\nSalvo em {out}/ (best_genXXX.npy, history.csv, fitness.png)")


if __name__ == "__main__":
    main()
