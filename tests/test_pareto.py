"""Tests for Pareto utilities."""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.pareto import (
    dominates, non_dominated_sort, crowding_distance,
    hypervolume_2d, igd, unary_epsilon, merge_archives,
)


def test_dominates_basic():
    assert dominates((10, 10), (20, 20))
    assert dominates((10, 10), (10, 20))
    assert dominates((10, 10), (20, 10))
    assert not dominates((10, 10), (10, 10))
    assert not dominates((10, 20), (20, 10))
    assert not dominates((20, 20), (10, 10))


def test_non_dominated_sort():
    points = [(10, 20), (20, 10), (15, 15), (25, 25), (5, 30)]
    fronts = non_dominated_sort(points)

    # First front should be the non-dominated ones
    assert len(fronts[0]) > 0
    # (25, 25) should NOT be in first front
    first_front_indices = fronts[0]
    assert 3 not in first_front_indices  # (25, 25) is dominated


def test_crowding_distance():
    points = [(10, 20), (20, 10), (15, 15)]
    indices = [0, 1, 2]
    cd = crowding_distance(points, indices)

    # Boundary points should have inf distance
    # With 3 points, the middle one has finite distance
    assert all(v >= 0 for v in cd.values())


def test_hypervolume_2d():
    points = [(0, 10), (5, 5), (10, 0)]
    ref = (15, 15)
    hv = hypervolume_2d(points, ref)
    assert hv > 0


def test_hypervolume_empty():
    assert hypervolume_2d([], (10, 10)) == 0.0


def test_igd_basic():
    candidate = [(10, 10), (20, 5)]
    reference = [(10, 10)]
    assert igd(candidate, reference) == pytest.approx(0.0, abs=0.01)


def test_igd_empty_candidate():
    assert igd([], [(10, 10)]) == float("inf")


def test_merge_archives():
    a = [((10, 20), {"a": 1}), ((20, 10), {"a": 2})]
    b = [((15, 15), {"b": 1}), ((25, 25), {"b": 2})]
    merged = merge_archives(a, b, capacity=4)
    # (25, 25) should be dominated and removed
    assert len(merged) <= 4
    assert not any(m[0] == (25, 25) for m in merged)
