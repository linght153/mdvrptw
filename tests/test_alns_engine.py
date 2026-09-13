"""Tests for the净化后的 ALNS engine (standard operators only, no learning selector).

迁移自旧项目 test_alns_engine.py 的行为契约，但仅保留标准引擎行为：
- 引擎可运行并产出 Pareto 档案
- 档案只含可行正目标
- 轨迹日志 JSON-safe
- archive checkpoint 语义（初始/间隔/最终）
- 上下文构造（SearchContext 边界校验）
- 段奖励选择器与引擎集成
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.instance import load_instance
from src.core.initial_solution import nearest_neighbor
from src.core.objectives import calculate_objectives
from src.alns.engine import ALNSEngine
from src.alns.selection_context import SearchContext


@pytest.fixture
def instance_p01():
    return load_instance("p01")


def _base_config(**overrides):
    config = {
        "max_iterations": 50,
        "max_time_seconds": 30,
        "stagnation_limit": 20,
        "segment_size": 10,
        "initial_temperature": 100.0,
        "cooling_rate": 0.999,
        "archive_capacity": 20,
        "parent_selection": "crowding",
        "initial_solution": "nearest",
        "k_min_ratio": 0.10,
        "k_max_ratio": 0.40,
        "sigma": 10.0,
    }
    config.update(overrides)
    return config


def test_engine_runs(instance_p01):
    """Engine should complete a run without errors."""
    engine = ALNSEngine(instance_p01, _base_config(), np.random.default_rng(42))
    archive = engine.run()
    assert archive.size >= 1
    assert len(engine.trajectory) > 0


def test_archive_grows_with_feasible_positive_objectives(instance_p01):
    """Archive should contain only feasible solutions with positive objectives."""
    config = _base_config(max_iterations=100, stagnation_limit=50)
    engine = ALNSEngine(instance_p01, config, np.random.default_rng(42))
    archive = engine.run()
    assert archive.size >= 1
    for obj, _ in archive.entries:
        assert obj[0] > 0
        assert obj[1] > 0


def test_trajectory_logging_is_json_safe(instance_p01):
    """Trajectory should be serializable without NaN."""
    config = _base_config(
        max_iterations=30,
        parent_selection="random",
    )
    engine = ALNSEngine(instance_p01, config, np.random.default_rng(42))
    engine.run()
    assert len(engine.trajectory) > 0
    json.dumps(engine.trajectory, allow_nan=False)


def test_trajectory_records_operator_pair(instance_p01):
    """Trajectory rows should record destroy/repair names and outcome."""
    engine = ALNSEngine(instance_p01, _base_config(), np.random.default_rng(42))
    engine.run()
    row = engine.trajectory[0]
    action = row["action"]
    assert action["destroy_name"] in engine.destroy_names
    assert action["repair_name"] in engine.repair_names
    assert "outcome" in row


def test_initial_archive_checkpoint_records_only_initial_objective(instance_p01):
    """Checkpoint at iteration 0 records only the initial solution objective."""
    initial_solution = nearest_neighbor(instance_p01, np.random.default_rng(99))
    expected_objective = calculate_objectives(initial_solution)
    engine = ALNSEngine(
        instance_p01,
        _base_config(
            max_iterations=1,
            archive_checkpoint_interval=10,
            local_search_freq=0,
            local_search_on_accept=False,
            local_search_final=False,
        ),
        np.random.default_rng(42),
        initial_solution=initial_solution,
    )
    engine.run()
    assert engine.initial_archive_checkpoint == {
        "iteration": 0,
        "elapsed_seconds": 0.0,
        "objectives": [[float(expected_objective[0]), float(expected_objective[1])]],
    }
    assert all(item["iteration"] > 0 for item in engine.archive_checkpoints)


def test_archive_checkpoints_include_intervals_and_unique_final(instance_p01):
    """Checkpoint iterations should be [interval, 2*interval, ..., final]."""
    config = _base_config(
        max_iterations=25,
        stagnation_limit=100,
        archive_checkpoint_interval=10,
        parent_selection="random",
        local_search_freq=0,
        local_search_on_accept=False,
        local_search_final=False,
    )
    engine = ALNSEngine(instance_p01, config, np.random.default_rng(42))
    engine.run()
    assert [item["iteration"] for item in engine.archive_checkpoints] == [10, 20, 25]
    assert all(
        set(item) == {"iteration", "elapsed_seconds", "objectives"}
        for item in engine.archive_checkpoints
    )
    elapsed = [item["elapsed_seconds"] for item in engine.archive_checkpoints]
    assert all(math.isfinite(value) and value >= 0.0 for value in elapsed)
    assert elapsed == sorted(elapsed)
    json.dumps(engine.archive_checkpoints, allow_nan=False)


def test_archive_checkpoint_interval_rejects_invalid_values(instance_p01):
    """Non-integer or negative checkpoint interval should raise."""
    for bad in [-1, 1.5, True, "10"]:
        with pytest.raises(ValueError):
            ALNSEngine(
                instance_p01,
                _base_config(archive_checkpoint_interval=bad),
                np.random.default_rng(42),
            )


def test_zero_iteration_run_keeps_initial_archive_out_of_checkpoint_history(instance_p01):
    """With max_iterations=0, no checkpoint rows but initial checkpoint present."""
    engine = ALNSEngine(
        instance_p01,
        _base_config(
            max_iterations=0,
            archive_checkpoint_interval=10,
            local_search_freq=0,
            local_search_on_accept=False,
            local_search_final=False,
        ),
        np.random.default_rng(42),
    )
    engine.run()
    assert engine.archive_checkpoints == []
    assert engine.initial_archive_checkpoint is not None


def test_search_context_fields_are_bounded(instance_p01):
    """SearchContext should enforce bounded ratios."""
    config = _base_config(max_iterations=100)
    engine = ALNSEngine(instance_p01, config, np.random.default_rng(42))
    engine.run()
    # 构造一个非法上下文应被拒绝
    with pytest.raises(ValueError):
        SearchContext(
            iteration=0,
            progress=1.5,  # > 1.0
            stagnation_ratio=0.5,
            temperature_ratio=0.5,
            recent_acceptance_rate=0.5,
            recent_archive_add_rate=0.5,
            archive_size_ratio=0.5,
            capacity_utilization_mean=0.5,
            capacity_utilization_std=0.1,
            recent_improvement=0.5,
        )


def test_engine_uses_segment_reward_selection_by_default(instance_p01):
    """Default selector should be SegmentRewardSelection."""
    engine = ALNSEngine(instance_p01, _base_config(), np.random.default_rng(42))
    assert type(engine.selector).__name__ == "SegmentRewardSelection"
