"""Políticas de referência (sem aprendizado) para validar a pista e comparar depois.

Se nem uma heurística simples consegue completar a pista, o problema é a pista
ou a física, não a evolução. Por isso ela já aparece na Fase 1.
"""

from __future__ import annotations

import numpy as np

from carros.sim.geometry import FloatArray


def heuristic_policy(sensors: FloatArray, offsets: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Vira para o lado com mais espaço e freia quando a frente está perto.

    Args:
        sensors: ``(N, R)`` leituras normalizadas.
        offsets: ``(R,)`` ângulos dos raios relativos à frente.

    Returns:
        ``(steer, throttle)`` arrays ``(N,)``.
    """
    steer = np.clip(3.0 * (sensors * np.sin(offsets)).sum(axis=1), -1.0, 1.0)
    front = sensors[:, len(offsets) // 2]
    throttle = np.clip(3.0 * front - 1.0, -1.0, 1.0)
    return steer, throttle


def random_policy(n: int, rng: np.random.Generator) -> tuple[FloatArray, FloatArray]:
    """Comandos aleatórios uniformes em ``[-1, 1]``."""
    return rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)
