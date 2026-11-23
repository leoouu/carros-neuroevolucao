"""Gera pistas aleatórias e salva cada uma como YAML.

Exemplo:
    python scripts/generate_tracks.py --out configs/tracks/train --prefix train --seeds 0-7
    python scripts/generate_tracks.py --out configs/tracks/unseen --prefix unseen --seeds 1000-1007
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from carros.sim.generator import generate_track_dict, load_generator_params


def parse_seeds(text: str) -> list[int]:
    """``"0-3,10"`` -> ``[0, 1, 2, 3, 10]``."""
    seeds: list[int] = []
    for part in text.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return seeds


def main() -> None:
    """Gera e grava as pistas."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--params", default="configs/generator.yaml")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--prefix", default="gen")
    p.add_argument("--seeds", required=True, help='ex.: "0-7" ou "1,5,9"')
    args = p.parse_args()

    params = load_generator_params(args.params)
    args.out.mkdir(parents=True, exist_ok=True)
    for seed in parse_seeds(args.seeds):
        name = f"{args.prefix}_{seed:04d}"
        data = generate_track_dict(name, params, seed)
        header = f"# Gerada por scripts/generate_tracks.py (seed {seed}, {args.params}).\n"
        text = header + yaml.safe_dump(data, sort_keys=False, default_flow_style=None)
        (args.out / f"{name}.yaml").write_text(text, encoding="utf-8")
        print(f"{args.out / name}.yaml")


if __name__ == "__main__":
    main()
