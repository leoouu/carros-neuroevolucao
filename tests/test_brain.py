"""Rede neural em lote e serialização."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from carros.brain.mlp import MLPSpec, forward, load_genome, random_genomes, save_genome


def test_param_count() -> None:
    assert MLPSpec((8, 8, 2)).n_params == 8 * 8 + 8 + 8 * 2 + 2
    assert MLPSpec((3, 4, 5, 2)).n_params == (12 + 4) + (20 + 5) + (10 + 2)


def test_invalid_spec_rejected() -> None:
    with pytest.raises(ValueError):
        MLPSpec((8,))


def test_forward_matches_manual_computation() -> None:
    spec = MLPSpec((2, 3, 1))
    rng = np.random.default_rng(0)
    g = random_genomes(spec, 1, rng, 1.0)
    x = np.array([[0.3, -0.7]])
    w1 = g[0, :6].reshape(2, 3)
    b1 = g[0, 6:9]
    w2 = g[0, 9:12].reshape(3, 1)
    b2 = g[0, 12:13]
    expected = np.tanh(np.tanh(x @ w1 + b1) @ w2 + b2)
    np.testing.assert_allclose(forward(spec, g, x), expected)


def test_each_car_uses_its_own_genome() -> None:
    spec = MLPSpec((4, 5, 2))
    rng = np.random.default_rng(1)
    genomes = random_genomes(spec, 6, rng, 1.0)
    x = rng.random((6, 4))
    batch = forward(spec, genomes, x)
    one_by_one = np.vstack([forward(spec, genomes[i : i + 1], x[i : i + 1]) for i in range(6)])
    np.testing.assert_allclose(batch, one_by_one)
    assert np.all(np.abs(batch) <= 1.0)


def test_wrong_genome_size_rejected() -> None:
    with pytest.raises(ValueError):
        forward(MLPSpec((2, 2)), np.zeros((1, 5)), np.zeros((1, 2)))


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    spec = MLPSpec((8, 8, 2))
    g = random_genomes(spec, 1, np.random.default_rng(2), 1.0)[0]
    save_genome(tmp_path / "best_gen007", g, spec, {"generation": 7, "fitness": 3.5})
    g2, spec2, meta = load_genome(tmp_path / "best_gen007.npy")
    np.testing.assert_array_equal(g, g2)
    assert spec2 == spec
    assert meta == {"generation": 7, "fitness": 3.5}
