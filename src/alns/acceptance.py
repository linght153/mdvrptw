"""基于 Pareto 支配的多目标 ALNS 接受准则。

SA 温度自动校准, 参考:
  - Pisinger & Ropke (2007): T₀ = -w·f(s₀)/ln(0.5)
  - Ropke & Pisinger (2006): 自适应冷却率
"""

import math

from src.core.pareto import Objective, dominates


class ParetoAcceptance:
    """
    基于 Pareto 支配接受新解，对严格被支配解采用模拟退火。

    决策规则:
      - 新解支配当前解 → 接受
      - 当前解支配新解 → 以概率 exp(-delta/T) 接受
      - 互不支配 → 接受（鼓励多样性）

    SA 参数:
      T₀ 自动校准: T₀ = -w·f(s₀)/ln(0.5), 使 w=5% 劣化解有 50% 接受概率
      冷却率自适应: c = (T_end/T₀)^(1/N), 保证 N 次迭代后 T_end ≈ T₀·1e-4

    也可通过 initial_temperature/cooling_rate 显式覆盖。
    """

    def __init__(
        self,
        initial_temperature: float | None = None,
        cooling_rate: float | None = None,
        max_iterations: int = 5000,
        initial_objective: Objective | None = None,
        t0_ratio: float = 0.05,          # w: 初始目标值的百分比
        t_end_ratio: float = 1e-4,        # T_end / T₀
    ):
        # ── 校准 T₀ ──────────────────────────────────────────
        if initial_temperature is not None:
            self.T0 = float(initial_temperature)
        elif initial_objective is not None:
            f0 = initial_objective[0] + initial_objective[1]
            self.T0 = -t0_ratio * f0 / math.log(0.5)  # ≈ 0.0722 * w * f₀
        else:
            self.T0 = 50.0  # 安全回退

        self.T = self.T0

        # ── 校准冷却率 ────────────────────────────────────────
        self._explicit_cooling = cooling_rate is not None
        if cooling_rate is not None:
            self.cooling_rate = float(cooling_rate)
        elif max_iterations > 0:
            # c^N × T₀ = t_end_ratio × T₀  →  c = exp(ln(t_end_ratio)/N)
            self.cooling_rate = math.exp(math.log(t_end_ratio) / max_iterations)
        else:
            self.cooling_rate = 0.999

        # 存储用于日志
        self.max_iterations = max_iterations
        self._t_end_ratio = t_end_ratio
        # 是否需要用初始解自动校准（未显式提供温度/初始目标时为 True）
        self._auto_calibrate = (initial_temperature is None and initial_objective is None)

    def calibrate(self, initial_objective: Objective, max_iterations: int):
        """用实际初始解和目标迭代次数重新校准温度（外部调用）。"""
        f0 = initial_objective[0] + initial_objective[1]
        self.T0 = -0.05 * f0 / math.log(0.5)
        self.T = self.T0
        self.max_iterations = max_iterations
        if not self._explicit_cooling:
            # 显式冷却率 (逃逸实验 V3) 不被自动校准覆盖
            self.cooling_rate = math.exp(math.log(self._t_end_ratio) / max(max_iterations, 1))
        self.T = self.T0
        self._auto_calibrate = False

    def reset(self):
        self.T = self.T0

    def accept(
        self,
        current_obj: Objective,
        new_obj: Objective,
        rng,
    ) -> bool:
        """
        判断是否接受新解作为下一父代。
        """
        if dominates(new_obj, current_obj):
            return True
        if dominates(current_obj, new_obj):
            # 模拟退火: 使用绝对组合目标差值
            delta = (new_obj[0] + new_obj[1]) - (current_obj[0] + current_obj[1])
            if delta <= 0:
                return True  # 不应发生, 但安全处理
            if self.T > 1e-10:
                prob = math.exp(-delta / self.T)
                return rng.random() < prob
            return False
        # 互不支配：接受以促进多样性
        return True

    def cool(self):
        """降温。"""
        self.T *= self.cooling_rate
