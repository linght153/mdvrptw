"""种群混合交叉算子 — route-copy (A 臂) 与 partition (B 臂)。

- route_copy_crossover (2026-09-04): 后代 = 双亲路由的混合继承 — 父代
  A/B 路由各自随机序 × 继承率 (车场车辆数上限内, 且不覆盖已继承客户),
  未覆盖客户 TW-aware 修复, 返回完整可行解 (可行性兜底由 engine 公共链
  保证)。
- partition_crossover (2026-09-07, M2): 车场簇继承 — load-bearing 单位 =
  车场分配簇。按车场固定序逐车场掷硬币, 整车场路由组从所选父本继承;
  修复池大小 (= 跨父本重复客户数) 如实暴露供 cx_duplicates 统计。
"""
import numpy as np

from src.core.solution import Solution


def route_copy_crossover(
    a_sol: Solution,
    b_sol: Solution,
    rng: np.random.Generator,
    inherit_prob: float = 0.6,
) -> Solution:
    """从两个父代生成一个后代。

    每车场继承路由数 ≤ vehicles_available; 冲突/未继承路由的客户进修复池。
    """
    inst = a_sol.instance
    nd = inst.num_depots
    avail = {d: inst.depots[d].vehicles_available for d in range(nd)}

    child = Solution(inst)  # routes: 每车场空列表
    covered: set[int] = set()
    pool: list[int] = []

    def try_add(di: int, route: list[int]) -> bool:
        """继承一条路由: 车场未满且与已继承客户无冲突才成功。"""
        if len(child.routes[di]) >= avail[di]:
            return False
        if any(c in covered for c in route):
            return False
        child.routes[di].append(list(route))
        covered.update(route)
        return True

    for parent in (a_sol, b_sol):
        routes = [(di, r) for di, rl in parent.routes.items() for r in rl]
        rng.shuffle(routes)  # 继承顺序随机化 (同一父代多解稳定性)
        for di, route in routes:
            if rng.random() < inherit_prob:
                if not try_add(di, route):
                    pool.extend(route)
            else:
                pool.extend(route)

    unassigned = sorted(set(pool) - covered)
    if unassigned:
        if inst.has_time_windows:
            from src.operators.repair import greedy_cost_tw_insertion as repair
        else:
            from src.operators.repair import greedy_cost_insertion as repair
        child = repair(child, unassigned, rng)

    # 多车场不变量同步 (repair 已同步, 继承路由部分也需对齐)
    from src.operators.repair import _sync_assigned_depots
    _sync_assigned_depots(child)
    return child


def partition_crossover(a_sol: Solution, b_sol: Solution, rng) -> Solution:
    """车场簇继承交叉 (M2, 2026-09-07) — 池内重组 B 臂。

    判别单位 = 车场分配簇 (vs route_copy 的路由级 A 臂)。机制:
    1. 对车场 d 按固定序 0..nd-1 逐车场掷硬币: rng.random() < 0.5 → 父本 A
       否则 B (每车场独立; rng 消耗与父本结构无关, 确定性)。
    2. 该车场在所选父本的全部路由逐条复制入子代 (路由序保持); 已覆盖客户
       (先前车场已复制 = 跨父本重复) 跳过进修复池; 复制不拆路由 — 过滤重复
       客户后的子路由保原序, 非空仍整条保留。
    3. 修复池客户 (硬币交叉丢失的未覆盖客户) 用既有 repair 重插 (TW 实例
       greedy_cost_tw_insertion, 否则 greedy_cost_insertion, 与 route_copy
       同款) — 允许跨车场。
    4. _sync_assigned_depots 收尾。

    设计注 (M2 规格): 修复池大小 = 跨父本重复客户数 — 两父本分配结构越
    接近, 重复越少 (继承纯度越高); A/B 车场偏好相反的父本组合重复多 →
    经 child._cx_duplicates 如实暴露, engine 累计进 cx_duplicates, verdict 用。
    返回完整解 (重复跳过后的修复池只作计数; 真正未覆盖客户由 repair 补全)。
    """
    inst = a_sol.instance
    nd = inst.num_depots

    child = Solution(inst)
    covered: set[int] = set()
    duplicates: list[int] = []  # 跨父本重复 (已覆盖) 客户 — 修复池计数

    # 1+2. 车场固定序逐车场掷硬币 + 继承所选父本的车场全部路由
    for d in range(nd):
        parent = a_sol if rng.random() < 0.5 else b_sol
        for route in parent.routes[d]:  # 路由序保持
            kept = []
            for c in route:
                if c in covered:
                    duplicates.append(c)  # 已覆盖 → 跳过进修复池计数
                else:
                    covered.add(c)
                    kept.append(c)
            if kept:  # 过滤重复后非空仍整条保留 (原序)
                child.routes[d].append(kept)

    # 3. 修复池客户重插: 只补真正未覆盖 (重复客户已在子代覆盖集内, 不重插)
    unassigned = sorted(child.unassigned_customers())
    if unassigned:
        if inst.has_time_windows:
            from src.operators.repair import greedy_cost_tw_insertion as repair
        else:
            from src.operators.repair import greedy_cost_insertion as repair
        child = repair(child, unassigned, rng)

    # 4. 多车场不变量同步 (route_copy 同款收尾)
    from src.operators.repair import _sync_assigned_depots
    _sync_assigned_depots(child)

    child._cx_duplicates = len(duplicates)
    return child
