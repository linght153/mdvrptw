"""算子选择器共享的搜索上下文和反馈数据。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchContext:
    """一次算子选择前可观测且归一化的搜索状态。"""

    iteration: int
    progress: float
    stagnation_ratio: float
    temperature_ratio: float
    recent_acceptance_rate: float
    recent_archive_add_rate: float
    archive_size_ratio: float
    capacity_utilization_mean: float
    capacity_utilization_std: float
    recent_improvement: float

    def __post_init__(self):
        bounded = {
            "progress": self.progress,
            "stagnation_ratio": self.stagnation_ratio,
            "temperature_ratio": self.temperature_ratio,
            "recent_acceptance_rate": self.recent_acceptance_rate,
            "recent_archive_add_rate": self.recent_archive_add_rate,
            "archive_size_ratio": self.archive_size_ratio,
            "capacity_utilization_mean": self.capacity_utilization_mean,
        }
        for name, value in bounded.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.iteration < 0:
            raise ValueError("iteration must be nonnegative")
        if self.capacity_utilization_std < 0.0:
            raise ValueError("capacity_utilization_std must be nonnegative")


@dataclass(frozen=True)
class OperatorOutcome:
    """一次破坏—修复算子对执行后的统一反馈。"""

    accepted: bool
    archive_added: bool
    dominates_parent: bool
    feasible: bool
    crashed: bool
    score: float
    reward: float
    elapsed_seconds: float
    destroy_crashed: bool = False
    repair_crashed: bool = False

    def __post_init__(self):
        if self.elapsed_seconds < 0.0:
            raise ValueError("elapsed_seconds must be nonnegative")
