"""ALNS 算子选择策略。"""

from typing import Protocol

import numpy as np

from src.alns.selection_context import OperatorOutcome, SearchContext


class OperatorPairSelector(Protocol):
    """共享ALNS引擎使用的算子对选择契约。"""

    def select_pair(
        self,
        context: SearchContext,
        rng: np.random.Generator,
    ) -> tuple[int, int]: ...

    def update(
        self,
        action: tuple[int, int],
        outcome: OperatorOutcome,
        next_context: SearchContext,
    ) -> None: ...

    def get_diagnostics(self) -> dict: ...

    def get_last_decision(self) -> dict | None: ...


class SegmentRewardSelection:
    """
    标准 ALNS 基于段的奖励机制 (Pisinger & Ropke 2007)。

    算子通过轮盘赌选择。其性能按段追踪：每 `segment_size` 次迭代后，
    根据累积奖励更新权重。

    与原文一致的关键参数:
      - reaction_factor = 0.1 (原文推荐, 曾错误使用 0.5)
      - decay_factor = 1.0 (无全局衰减; 原文无此概念, 曾错误使用 0.95)
      - min_selection_prob = 0.005 (最低选择概率, 保证探索)
    """

    def __init__(self, destroy_names: list[str], repair_names: list[str],
                 segment_size: int = 100,
                 reaction_factor: float = 0.1, decay_factor: float = 1.0,
                 min_selection_prob: float = 0.005):
        self.destroy_names = list(destroy_names)
        self.repair_names = list(repair_names)
        self.names = self.destroy_names + self.repair_names
        self.segment_size = segment_size
        self.reaction_factor = reaction_factor
        self.decay_factor = decay_factor
        self.min_selection_prob = min_selection_prob

        # 权重（初始均等）
        self.weights = np.ones(len(self.names))

        # 当前段的分数累加器
        self.scores = np.zeros(len(self.names))
        self.usage_count = np.zeros(len(self.names), dtype=int)

        self.iter_in_segment = 0

    def select(self, rng: np.random.Generator) -> int:
        """按权重轮盘赌选择算子索引，保证最低选择概率。"""
        w = np.maximum(self.weights, 1e-6)
        probs = w / w.sum()
        # 保证所有算子的最低选择概率
        n = len(probs)
        min_p = self.min_selection_prob
        probs = np.clip(probs, min_p, None)
        probs = probs / probs.sum()
        return int(rng.choice(n, p=probs))

    def _sample_range(
        self, start: int, stop: int, rng: np.random.Generator,
    ) -> int:
        """仅在指定算子类别内按权重采样。"""
        weights = np.maximum(self.weights[start:stop], 1e-6)
        probs = weights / weights.sum()
        probs = np.clip(probs, self.min_selection_prob, None)
        probs /= probs.sum()
        return start + int(rng.choice(stop - start, p=probs))

    def select_pair(
        self, context: SearchContext, rng: np.random.Generator,
    ) -> tuple[int, int]:
        """分别在破坏与修复算子类别内选择一个索引。"""
        d_idx = self._sample_range(0, len(self.destroy_names), rng)
        offset = len(self.destroy_names)
        global_r_idx = self._sample_range(offset, len(self.names), rng)
        return d_idx, global_r_idx - offset

    def update(
        self,
        action: tuple[int, int],
        outcome: OperatorOutcome,
        next_context: SearchContext,
    ):
        """使用一次算子对执行结果更新两个个体算子的统计。"""
        d_idx, r_idx = action
        self.update_score(
            d_idx, 0.0 if outcome.destroy_crashed else outcome.score,
        )
        self.update_score(
            len(self.destroy_names) + r_idx,
            0.0 if outcome.repair_crashed else outcome.score,
        )
        self.step()

    def get_diagnostics(self) -> dict:
        return {
            "weights": self.weights.copy(),
            "usage_count": self.usage_count.copy(),
        }

    def get_last_decision(self) -> dict | None:
        return None

    def update_score(self, operator_idx: int, score: float):
        """
        累加算子奖励。

        分数等级：
          0: 解被拒绝
          1: 解被接受（新的非支配解）
          2: 解被接受（支配当前父代）
          3: 解被接受（新的全局最优）
        """
        self.scores[operator_idx] += score
        self.usage_count[operator_idx] += 1

    def end_segment(self):
        """段结束时更新权重。

        遵循 Pisinger & Ropke (2007) 公式:
          w_{i,j+1} = w_{i,j} · (1 - r) + r · (π_i / θ_i)
        其中 r = reaction_factor, π_i = 累积分数, θ_i = 使用次数。
        """
        for i in range(len(self.names)):
            if self.usage_count[i] > 0:
                avg_score = self.scores[i] / self.usage_count[i]
                self.weights[i] = (
                    self.weights[i] * (1 - self.reaction_factor)
                    + self.reaction_factor * avg_score
                )
            # 全局衰减 (仅当 decay_factor < 1 时生效)
            if self.decay_factor < 1.0:
                self.weights[i] *= self.decay_factor
            # 软下限：防止权重坍缩为零, 但远低于之前 0.1 以避免抹平差异
            self.weights[i] = max(self.weights[i], self.min_selection_prob * 2)

        self.scores = np.zeros(len(self.names))
        self.usage_count = np.zeros(len(self.names), dtype=int)
        self.iter_in_segment = 0

    def penalize_crash(self, name: str):
        """惩罚崩溃的算子：将其权重降至最低，减少后续被选中的概率。"""
        if name in self.names:
            idx = self.names.index(name)
            # 大幅削减权重（但保留 min_selection_prob 探索机会）
            self.weights[idx] = max(self.weights[idx] * 0.3, self.min_selection_prob)
            # 也重置当前段分数，避免 crash 算子因旧分数获得奖励
            self.scores[idx] = max(0, self.scores[idx] - 1.0)

    def is_effectively_dead(self, name: str) -> bool:
        """检查算子是否被有效禁用（权重降至最低水平）。"""
        if name in self.names:
            idx = self.names.index(name)
            return self.weights[idx] <= self.min_selection_prob * 2
        return False

    def step(self):
        """迭代计数器递增；达到段大小时自动结束段。"""
        self.iter_in_segment += 1
        if self.iter_in_segment >= self.segment_size:
            self.end_segment()


class UniformRandomSelection:
    """均匀随机算子选择（用于 Random ALNS 基线）。"""

    def __init__(self, destroy_names: list[str], repair_names: list[str]):
        self.destroy_names = list(destroy_names)
        self.repair_names = list(repair_names)
        self.names = self.destroy_names + self.repair_names
        self.usage_count = np.zeros(len(self.names), dtype=int)

    def select(self, rng: np.random.Generator) -> int:
        return int(rng.integers(0, len(self.names)))

    def select_pair(
        self, context: SearchContext, rng: np.random.Generator,
    ) -> tuple[int, int]:
        return (
            int(rng.integers(0, len(self.destroy_names))),
            int(rng.integers(0, len(self.repair_names))),
        )

    def update(
        self,
        action: tuple[int, int],
        outcome: OperatorOutcome,
        next_context: SearchContext,
    ):
        d_idx, r_idx = action
        self.usage_count[d_idx] += 1
        self.usage_count[len(self.destroy_names) + r_idx] += 1

    def get_diagnostics(self) -> dict:
        return {"usage_count": self.usage_count.copy()}

    def get_last_decision(self) -> dict | None:
        return None

    def update_score(self, operator_idx: int, score: float):
        pass

    def end_segment(self):
        pass

    def step(self):
        pass
