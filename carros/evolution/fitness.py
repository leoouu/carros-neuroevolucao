"""Função de fitness: o que a evolução vai maximizar.

    fitness = progresso + bônus_de_volta * completou - penalidade * tempo_vivo

* **progresso** (checkpoints + fração até o próximo) é o termo principal. Ele é
  contínuo, então já na geração 1 há diferença entre "bateu logo" e "bateu
  quase no checkpoint" — a seleção tem por onde puxar.
* **bônus de volta** grande: completar a volta tem que valer mais que qualquer
  volta incompleta.
* **penalidade leve de tempo**: entre dois carros que completam, o mais rápido
  vence (ele fica menos tempo "vivo" até cruzar a chegada). É leve de propósito:
  um checkpoint vale muito mais que alguns segundos, então ninguém ganha batendo cedo.

Não recompensamos velocidade diretamente: isso ensina os carros a bater rápido.
"""

from __future__ import annotations

import numpy as np

from carros.evolution.config import FitnessParams
from carros.sim.geometry import BoolArray, FloatArray


def compute_fitness(
    progress: FloatArray, finished: BoolArray, time_alive: FloatArray, params: FitnessParams
) -> FloatArray:
    """Fitness ``(P,)`` de cada carro a partir das estatísticas do episódio."""
    result: FloatArray = (
        progress + params.lap_bonus * finished.astype(np.float64) - params.time_penalty * time_alive
    )
    return result
