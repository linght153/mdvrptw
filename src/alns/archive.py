"""Pareto 最优解的外部精英档案。"""

import json
from pathlib import Path

from src.core.pareto import (
    Objective,
    dominates,
    crowding_distance,
    non_dominated_sort,
)


class ParetoArchive:
    """
    维护一组容量控制的非支配解。

    每个条目为 (目标值, 信息)，其中信息字典包含
    额外数据（如解表示、迭代次数）。
    """

    def __init__(self, capacity: int = 80, single_objective: bool = False,
                 diversity: bool = False, diversity_threshold: float = 0.2,
                 fitness_dc: bool = False, fitness_kappa: float = 0.05,
                 admit_relaxed: bool = False):
        self.capacity = capacity
        self.single_objective = single_objective
        # 多样性感知裁剪 (探针, 2026-09-04): single_objective 满容时保 best +
        # 按路由 Jaccard 距离贪心保代表 (默认关 → 原 cost-only 裁剪逐位不变)
        self.diversity = diversity
        self.diversity_threshold = diversity_threshold
        # A2 (2026-09-04): 多样性贡献进 fitness — 裁 fitness = cost − κ×best×dc
        # 最大者 (默认关; 与 diversity 互斥, 优先级高于 diversity)
        self.fitness_dc = fitness_dc
        self.fitness_kappa = fitness_kappa
        # admit_only 判别臂 (2026-09-04): 只放宽准入 (满档不拒更差), 裁剪仍
        # cost-only — 隔离"准入放宽→stagnation 少重启"与"多样性裁剪"的效应
        self.admit_relaxed = admit_relaxed
        self.entries: list[tuple[Objective, dict]] = []

    @staticmethod
    def _route_keys(info: dict) -> frozenset:
        """条目路由键集: {(depot, tuple(seq))} (惰性缓存于 meta)。"""
        cached = info.get("_route_keys")
        if cached is not None:
            return cached
        sol = info.get("solution")
        if sol is None:
            return frozenset()
        keys = frozenset(
            (di, tuple(route))
            for di, rl in sol.routes.items()
            for route in rl
        )
        info["_route_keys"] = keys
        return keys

    @staticmethod
    def _jaccard_dist(ka: frozenset, kb: frozenset) -> float:
        if not ka or not kb:
            return 1.0
        union = len(ka | kb)
        if union == 0:
            return 0.0
        return 1.0 - len(ka & kb) / union

    def add(self, obj: Objective, info: dict) -> bool:
        """
        尝试将解加入档案。

        single_objective=True (MDVRPTW 单目标): emission 维恒 0, Pareto
        支配会拒掉所有更高 cost 的解 → 档案坍缩为单点 → 父代多样性崩
        (实测 pr20 2000iter gap 9.9% → 18.6%)。单目标档案 = 精英池:
        只拒绝近似重复 cost, 超容量按拥挤度 (一维 cost 间距) 裁剪。

        Returns:
            True 表示解被加入（即未被已有条目支配）
        """
        if self.single_objective:
            for existing_obj, _ in self.entries:
                if abs(existing_obj[0] - obj[0]) < 1e-6:
                    return False
            if not (self.diversity or self.fitness_dc or self.admit_relaxed):
                # 多样性/fitness/admit 模式放宽准入: 更差但结构不同的解也允许
                # 进池, 由裁剪决定去留 (否则多样代表永远进不来)
                if (len(self.entries) >= self.capacity
                        and obj[0] >= max(e[0][0] for e in self.entries)):
                    return False
            self.entries.append((obj, info))
            if len(self.entries) > self.capacity:
                if self.fitness_dc:
                    self._prune_single_fitness()
                elif self.diversity:
                    self._prune_single_diverse()
                else:
                    # 按 cost 保最优 capacity 个 (一维 crowding 的退化维会产生
                    # 伪 inf 边界, 曾裁掉全局最优解使 best 回升)
                    self.entries.sort(key=lambda e: e[0][0])
                    self.entries = self.entries[:self.capacity]
            return True

        # 检查是否被已有条目支配
        for existing_obj, _ in self.entries:
            if dominates(existing_obj, obj):
                return False

        # 拒绝近似重复的目标值（档案中已有相同值）
        for existing_obj, _ in self.entries:
            if abs(existing_obj[0] - obj[0]) < 1e-6 and abs(existing_obj[1] - obj[1]) < 1e-6:
                return False

        # 移除被新解支配的条目
        self.entries = [
            (e_obj, e_info)
            for e_obj, e_info in self.entries
            if not dominates(obj, e_obj)
        ]

        # 加入新条目
        self.entries.append((obj, info))

        # 超过容量时裁剪
        if len(self.entries) > self.capacity:
            self._prune()

        return True

    def _prune_single_diverse(self):
        """多样性感知裁剪 (single_objective + diversity=True)。

        cost 升序贪心: 保 best (首个必留), 与已保集合的路由 Jaccard 距离
        < threshold 的相似解跳过; 不足 capacity 时按 cost 从被跳过者补齐。
        输出按 cost 升序 (确定性)。O(P²) 每满容 add 一次 (P=capacity)。
        """
        ordered = sorted(self.entries, key=lambda e: e[0][0])
        kept: list[tuple[Objective, dict]] = []
        skipped: list[tuple[Objective, dict]] = []
        for e in ordered:
            if len(kept) >= self.capacity:
                skipped.append(e)
                continue
            if kept:
                keys_e = self._route_keys(e[1])
                min_d = min(self._jaccard_dist(keys_e, self._route_keys(k[1]))
                            for k in kept)
                if min_d < self.diversity_threshold:
                    skipped.append(e)
                    continue
            kept.append(e)
        if len(kept) < self.capacity:
            for e in skipped:
                if len(kept) >= self.capacity:
                    break
                kept.append(e)
        self.entries = kept

    def _prune_single_fitness(self):
        """多样性贡献进 fitness 的裁剪 (single_objective + fitness_dc=True)。

        dc = 条目到档案其余解的平均路由 Jaccard 距离 (∈[0,1]) —
        结构越新 dc 越大。fitness = cost − κ×best_cost×dc (dc 高 → fit 小
        → 优先保留)。满容时裁 fitness 最大者 = "成本高且结构重复" 先死;
        全局 best (最小 cost) 永不裁 (最终报告稳定)。输出按 cost 升序。
        O(P²) 每满容 add 一次 (P=capacity)。
        """
        n = len(self.entries)
        best_cost_v = min(e[0][0] for e in self.entries)
        keys = [self._route_keys(e[1]) for e in self.entries]
        # dc: 与其余条目的 Jaccard 距离均值
        dc = [0.0] * n
        for i in range(n):
            ds = [self._jaccard_dist(keys[i], keys[j])
                  for j in range(n) if j != i and keys[j]]
            if ds:
                dc[i] = sum(ds) / len(ds)
        fits = [e[0][0] - self.fitness_kappa * best_cost_v * dc[i]
                for i, e in enumerate(self.entries)]
        order = sorted(range(n), key=lambda i: fits[i])  # fit 小 = 好
        kept_idx = order[:self.capacity]
        best_idx = min(range(n), key=lambda i: self.entries[i][0][0])
        if best_idx not in kept_idx:
            kept_idx[-1] = best_idx  # 顶替 fit 最差者
        self.entries = [self.entries[i] for i in
                        sorted(kept_idx, key=lambda i: self.entries[i][0][0])]

    def _prune(self):
        """移除拥挤度最低的解。"""
        if len(self.entries) <= self.capacity:
            return

        objs = [e[0] for e in self.entries]
        indices = list(range(len(self.entries)))
        cd = crowding_distance(objs, indices)

        # 按拥挤度降序排序
        sorted_idx = sorted(indices, key=lambda i: cd.get(i, 0), reverse=True)
        keep = sorted_idx[:self.capacity]
        self.entries = [self.entries[i] for i in keep]

    @property
    def objectives(self) -> list[Objective]:
        return [e[0] for e in self.entries]

    @property
    def size(self) -> int:
        return len(self.entries)

    def is_empty(self) -> bool:
        return len(self.entries) == 0

    def best_cost(self) -> float:
        return min(e[0][0] for e in self.entries) if self.entries else float("inf")

    def best_emission(self) -> float:
        return min(e[0][1] for e in self.entries) if self.entries else float("inf")

    def clear(self):
        self.entries.clear()

    def save(self, filepath: str | Path):
        """将档案保存为 JSON。"""
        data = {
            "capacity": self.capacity,
            "size": len(self.entries),
            "entries": [
                {"cost": obj[0], "emission": obj[1], "info": info}
                for obj, info in self.entries
            ],
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str | Path):
        """从 JSON 加载档案。"""
        with open(filepath, "r") as f:
            data = json.load(f)
        self.capacity = data["capacity"]
        self.entries = [
            ((e["cost"], e["emission"]), e["info"])
            for e in data["entries"]
        ]

    def __repr__(self):
        return f"ParetoArchive(size={self.size}, capacity={self.capacity})"
