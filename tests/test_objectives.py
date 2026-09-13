"""Tests for objective computation."""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.instance import load_instance
from src.core.objectives import calculate_cost, calculate_emission, calculate_objectives
from src.core.initial_solution import nearest_neighbor
import numpy as np


@pytest.fixture
def instance_p01():
    return load_instance("p01")


@pytest.fixture
def solution_p01(instance_p01):
    rng = np.random.default_rng(42)
    return nearest_neighbor(instance_p01, rng)


def test_cost_positive(solution_p01):
    cost = calculate_cost(solution_p01)
    assert cost > 0, "Cost should be positive"


def test_emission_positive(solution_p01):
    emission = calculate_emission(solution_p01)
    assert emission > 0, "Emission should be positive"


def test_emission_proportional_to_cost(solution_p01):
    """Emission should be roughly proportional to cost * base_rate."""
    cost, emission = calculate_objectives(solution_p01)
    base_rate = solution_p01.instance.emission_model.base_emission_rate
    # With load correction, emission > cost * base_rate
    assert emission >= cost * base_rate * 0.9, (
        f"Emission ({emission}) should be >= cost * base_rate ({cost * base_rate})"
    )


def test_cost_unit_is_km(solution_p01):
    """With unit_cost=1.0, cost should equal total distance (手工重算, 修 2026-09-08 恒真断言)."""
    cost = calculate_cost(solution_p01)
    inst = solution_p01.instance
    expected = sum(
        inst.distance_matrix[inst.depots[d].index][route[0]]
        + inst.distance_matrix[route[-1]][inst.depots[d].index]
        + sum(inst.distance_matrix[route[i]][route[i + 1]]
              for i in range(len(route) - 1))
        for d, rl in solution_p01.routes.items()
        for route in rl
        if route
    )
    assert abs(cost - expected) < 1e-6
