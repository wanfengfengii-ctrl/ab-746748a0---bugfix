"""精确求解器：从可能含漏读（空格位）与杂点（划痕亮点）的标记坐标中恢复整数栅格。

联合优化：
  1. 原点 o、行基向量 b1、列基向量 b2（均在给定闭区间内，且 det(b1,b2) > 0）；
  2. 每个保留标记到栅格内互异格位 (r, c) 的分配（任意两个标记不得占用同一格位）；
  3. 弃点（杂点）数量不超过 max_outliers。

目标按字典序最小化：
  (弃点数, 最大曼哈顿残差, 残差总和, 完整参数序列, 完整分配序列)

搜索方式：六条闭区间跨度均不超过 6，基向量组合至多 7^4 个，先按行列式为正过滤，
再利用“标记必须能由 原点 = 坐标 - r*b1 - c*b2 落在原点区间”做快速预筛；对每组
参数，把互异分配建模成标记-格位二分图并用多项式算法精确求解：Kuhn 最大匹配定
最少弃点数，残差阈值二分定最小最大残差，再用一次匈牙利指派（整数费用同时编码
残差总和与完整分配序列）定剩余两级字典序目标。全部为整数运算，结果确定、可复现，
且在稠密同坐标标记下不会退化到指数级枚举。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

Cell = tuple[int, int]
DISCARD_CELL = (-1, -1)


def _kuhn_size(adj, stop_at=None):
    """二分图最大匹配（Kuhn 增广路）。adj[i] 为标记 i 可占用的格位下标。

    返回最多能互异分配的标记数；stop_at 给出后达到该值即可提前返回。
    """
    n = len(adj)
    owner: dict[int, int] = {}

    def augment(mi, seen):
        for cj in adj[mi]:
            if cj in seen:
                continue
            seen.add(cj)
            hold = owner.get(cj)
            if hold is None or augment(hold, seen):
                owner[cj] = mi
                return True
        return False

    size = 0
    for mi in sorted(range(n), key=lambda k: (len(adj[k]), k)):
        if adj[mi] and augment(mi, set()):
            size += 1
            if stop_at is not None and size >= stop_at:
                return size
    return size


def _hungarian(cost):
    """矩形最小费用指派（n <= m）：每个标记（行）分到互异列，返回 (行->列, 总费用)。

    允许大整数费用；不可行边应给出大于任何有限可行总费用的哨兵值，此时只有在
    不存在有限可行指派时才会选中不可行边。
    """
    n = len(cost)
    m = len(cost[0])
    inf = 1 + sum(max(row) for row in cost)
    u = [0] * (n + 1)
    v = [0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf
            j1 = 0
            row = cost[i0 - 1]
            for j in range(1, m + 1):
                if not used[j]:
                    cur = row[j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = [0] * n
    for j in range(1, m + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1
    total = sum(cost[i][assignment[i]] for i in range(n))
    return assignment, total


@dataclass(frozen=True)
class Marker:
    id: int
    x: int
    y: int


@dataclass
class _Solution:
    key: tuple
    assignment: dict[int, Cell]  # 标记下标（按 id 排序） -> (r, c)
    discarded: dict[int, str]    # 标记下标 -> 原因码


def solve(
    points,
    rows: int,
    cols: int,
    tolerance: int,
    origin_x,
    origin_y,
    basis_row_x,
    basis_row_y,
    basis_col_x,
    basis_col_y,
    max_outliers: int = 2,
):
    """执行复原。区间参数均为 (lo, hi) 闭区间，返回结果 dict。"""
    pts = [Marker(int(p["id"]), int(p["x"]), int(p["y"])) for p in points]
    pts.sort(key=lambda m: m.id)
    n = len(pts)
    budget = min(max_outliers, n - 1)
    need = n - budget  # 至少要保留的标记数

    # 行列式为正的基向量组合（与原点无关，先过滤）
    basis_pairs = []
    for b1x in range(basis_row_x[0], basis_row_x[1] + 1):
        for b1y in range(basis_row_y[0], basis_row_y[1] + 1):
            for b2x in range(basis_col_x[0], basis_col_x[1] + 1):
                for b2y in range(basis_col_y[0], basis_col_y[1] + 1):
                    if b1x * b2y - b1y * b2x > 0:
                        basis_pairs.append((b1x, b1y, b2x, b2y))

    stats = {
        "positive_determinant_basis_pairs": len(basis_pairs),
        "basis_pairs_rejected_pre_filter": 0,
        "parameter_sets_evaluated": 0,
        "complete_assignments_explored": 0,
    }

    result: dict = {"feasible": False}
    if not basis_pairs:
        result["reasons"] = [
            {
                "code": "no_positive_determinant_basis",
                "message": (
                    "给定的两条基向量闭区间内不存在行列式为正的整数组合，"
                    "无法构成行方向到列方向为逆时针的整数栅格"
                ),
            }
        ]
        result["stats"] = stats
        return result

    cells = [(r, c) for r in range(rows) for c in range(cols)]
    ox_lo, ox_hi = origin_x
    oy_lo, oy_hi = origin_y

    ever_fit = [False] * n          # 每个标记是否曾在某参数下落入某格位
    any_all_fit = False             # 是否存在所有标记单独都能落入格位的参数
    global_best: Optional[_Solution] = None

    def consider(ox, oy, b1x, b1y, b2x, b2y, cand, forced):
        """对一组参数求字典序最优互异分配并更新全局最优。

        forced 为完全没有容差内候选格位的标记下标集合（必须弃去）。
        把所有标记到“真实格位 + 弃点虚拟列”的指派一次性建模：
          1. Kuhn 最大匹配定最少弃点数；
          2. 残差阈值升序扫描定最小最大曼哈顿残差；
          3. 一次匈牙利指派，整数费用按
             “弃点数 ≫ 残差总和 ≫ 完整分配序列”分级编码，
             同时定最小残差总和与字典序最小分配序列。
        """
        nonlocal global_best, any_all_fit
        stats["parameter_sets_evaluated"] += 1
        if not forced:
            any_all_fit = True
        for i in range(n):
            if cand[i]:
                ever_fit[i] = True
        if len(forced) > budget:
            return

        ncells = rows * cols
        free_markers = [i for i in range(n) if cand[i]]
        # 每个可选标记的邻接表（格位下标）与残差
        adj: list[list[int]] = [[] for _ in range(n)]
        res_of: list[dict[int, int]] = [{} for _ in range(n)]
        for i in free_markers:
            for r, c, res in cand[i]:
                cj = r * cols + c
                adj[i].append(cj)
                res_of[i][cj] = res

        # 1) 最少弃点数：可互异放入真实格位的最大标记数
        matched = _kuhn_size(adj)
        d = n - matched
        if d > budget:
            return
        gb0 = global_best
        if gb0 is not None and d > gb0.key[0]:
            return

        # 2) 最小最大残差：在出现过的残差阈值上二分，找仍能互异放入
        #    matched 个标记的最小阈值。
        thresholds = sorted({res_of[i][cj] for i in free_markers for cj in adj[i]})
        best_max = 0

        def threshold_feasible(th):
            sub = [
                [cj for cj in adj[i] if res_of[i][cj] <= th] if cand[i] else []
                for i in range(n)
            ]
            return _kuhn_size(sub, stop_at=matched) >= matched

        if thresholds:
            if not threshold_feasible(thresholds[-1]):  # 理论上不会发生
                return
            lo, hi = 0, len(thresholds) - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if threshold_feasible(thresholds[mid]):
                    hi = mid
                else:
                    lo = mid + 1
            best_max = thresholds[lo]

        gb1 = global_best
        if gb1 is not None and d == gb1.key[0] and best_max > gb1.key[1]:
            return

        # 3a) 小整数费用匈牙利：只优化残差总和（弃点列为大常数 D0），
        #     得到本组参数下的最小残差总和，供与现任最优比较、剪枝。
        params_key = (ox, oy, b1x, b1y, b2x, b2y)
        D0 = n * (2 * tolerance + 1) + 1
        bad0 = D0 * (n + 1)
        ncol = ncells + d
        small = [[0] * ncol for _ in range(n)]
        for i in range(n):
            row = small[i]
            for cj in range(ncells):
                res = res_of[i].get(cj)
                if i in forced or res is None or res > best_max:
                    row[cj] = bad0
                else:
                    row[cj] = res
            for q in range(d):
                row[ncells + q] = D0
        col0, _ = _hungarian(small)
        total_res = 0
        for i, col in enumerate(col0):
            if col < ncells:
                total_res += res_of[i][col]

        # 残差总和不优于现任、且参数序列也不更小时，无资格成为全局最优
        gb2 = global_best
        if (
            gb2 is not None
            and d == gb2.key[0]
            and best_max == gb2.key[1]
            and (
                total_res > gb2.key[2]
                or (total_res == gb2.key[2] and params_key > gb2.key[3])
            )
        ):
            return

        # 3b) 分级整数费用匈牙利：在不改变弃点数/最大残差/残差总和的前提下
        #    取字典序最小的完整分配序列（大整数精确，无浮点）：
        #      真实列: res*B + q*W^(n-1-i)
        #      虚拟列: D     （弃点格位 (-1,-1) 编码数字 q=0）
        #    q=cj+1 按行主序与格位一一对应且严格递增；数字序列 (q_0..q_{n-1})
        #    的 W 进制位值之和，其数值大小顺序与“按标记编号排列的完整分配序列”
        #    字典序完全一致。B=W^n+1 大于任何位值之和，使残差总和优先；
        #    D 大于任何全真实费用，使弃点数最少优先。
        W = ncells + 1
        B = W ** n + 1
        D1 = (n * (2 * tolerance + 1) + 1) * B + 1
        bad1 = D1 * (n + 1)
        weights = [W ** (n - 1 - i) for i in range(n)]
        cost = [[0] * ncol for _ in range(n)]
        for i in range(n):
            row = cost[i]
            weight = weights[i]
            for cj in range(ncells):
                res = res_of[i].get(cj)
                if i in forced or res is None or res > best_max:
                    row[cj] = bad1
                else:
                    row[cj] = res * B + (cj + 1) * weight
            for q in range(d):
                row[ncells + q] = D1  # 弃点：位值数字 0

        col_for_row, _ = _hungarian(cost)

        assign: dict[int, Cell] = {}
        discarded: dict[int, str] = {}
        total_res = 0
        for i, col in enumerate(col_for_row):
            if col < ncells:
                r, c = divmod(col, cols)
                assign[i] = (r, c)
                total_res += res_of[i][col]
            else:
                discarded[i] = (
                    "no_cell_within_tolerance" if i in forced else "outlier_excluded"
                )
        assert len(discarded) == d
        seq = [assign.get(i, DISCARD_CELL) for i in range(n)]
        key = (d, best_max, total_res, params_key, tuple(seq))
        stats["complete_assignments_explored"] += 1
        if global_best is None or key < global_best.key:
            global_best = _Solution(key, assign, discarded)

    for b1x, b1y, b2x, b2y in basis_pairs:
        # centers[i] = [(r, c, cx, cy)]，cx/cy 使该标记落在 (r,c) 时
        # 原点应为 (cx, cy) = p - r*b1 - c*b2
        centers = []
        feasible_markers = 0
        for p in pts:
            lst = []
            feasible = False
            for r, c in cells:
                cx = p.x - r * b1x - c * b2x
                cy = p.y - r * b1y - c * b2y
                lst.append((r, c, cx, cy))
                # 存在原点区间内（含容差扩张）的 o 使该格位容纳此标记
                if (
                    ox_lo - tolerance <= cx <= ox_hi + tolerance
                    and oy_lo - tolerance <= cy <= oy_hi + tolerance
                ):
                    feasible = True
            centers.append(lst)
            if feasible:
                feasible_markers += 1

        if feasible_markers < need:
            stats["basis_pairs_rejected_pre_filter"] += 1
            for i in range(n):
                for _, _, cx, cy in centers[i]:
                    if (
                        ox_lo - tolerance <= cx <= ox_hi + tolerance
                        and oy_lo - tolerance <= cy <= oy_hi + tolerance
                    ):
                        ever_fit[i] = True
                        break
            continue
        for i in range(n):
            for _, _, cx, cy in centers[i]:
                if (
                    ox_lo - tolerance <= cx <= ox_hi + tolerance
                    and oy_lo - tolerance <= cy <= oy_hi + tolerance
                ):
                    ever_fit[i] = True
                    break

        for ox in range(ox_lo, ox_hi + 1):
            for oy in range(oy_lo, oy_hi + 1):
                cand = []
                forced = set()
                for i in range(n):
                    opts = []
                    for r, c, cx, cy in centers[i]:
                        dx = pts[i].x - (ox + r * b1x + c * b2x)
                        dy = pts[i].y - (oy + r * b1y + c * b2y)
                        if abs(dx) <= tolerance and abs(dy) <= tolerance:
                            opts.append((r, c, abs(dx) + abs(dy)))
                    if opts:
                        opts.sort(key=lambda t: (t[2], t[0], t[1]))
                    else:
                        forced.add(i)
                    cand.append(opts)
                consider(ox, oy, b1x, b1y, b2x, b2y, cand, forced)

    if global_best is None:
        reasons = []
        never = [pts[i].id for i in range(n) if not ever_fit[i]]
        if never and len(never) > max_outliers:
            reasons.append(
                {
                    "code": "too_many_unmatchable_markers",
                    "marker_ids": never,
                    "message": (
                        f"标记 {never} 在所有允许的原点/基向量下都无法落入容差范围内的"
                        f"任何格位；至少须弃去 {len(never)} 个，但杂点上限为 "
                        f"{max_outliers}，故无解"
                    ),
                }
            )
        elif never:
            reasons.append(
                {
                    "code": "markers_unmatchable_within_budget",
                    "marker_ids": never,
                    "message": (
                        f"标记 {never} 在所有允许的原点/基向量下都无法落入容差范围内的"
                        f"任何格位，必须作为杂点弃去（上限 {max_outliers} 个）"
                    ),
                }
            )
        if any_all_fit or not never:
            reasons.append(
                {
                    "code": "no_distinct_cell_assignment",
                    "message": (
                        "存在使每个标记单独都可落入格位的参数，但无法在弃点不超过 "
                        f"{max_outliers} 个的前提下把保留标记互异地分配到 "
                        f"{rows}×{cols} 栅格的不同格位（格位占用冲突无法消解），故无解"
                    ),
                }
            )
        elif not reasons:
            reasons.append(
                {
                    "code": "no_distinct_cell_assignment",
                    "message": (
                        "扣除必须弃去的标记后，其余标记仍无法互异分配到栅格格位，故无解"
                    ),
                }
            )
        result["reasons"] = reasons
        result["stats"] = stats
        return result

    ox, oy, b1x, b1y, b2x, b2y = global_best.key[3]
    det = b1x * b2y - b1y * b2x
    assignments = []
    discarded = []
    max_manhattan = 0
    total = 0
    for i, p in enumerate(pts):
        if i in global_best.assignment:
            r, c = global_best.assignment[i]
            px = ox + r * b1x + c * b2x
            py = oy + r * b1y + c * b2y
            dx = p.x - px
            dy = p.y - py
            man = abs(dx) + abs(dy)
            max_manhattan = max(max_manhattan, man)
            total += man
            assignments.append(
                {
                    "marker_id": p.id,
                    "grid_cell": [r, c],
                    "predicted": [px, py],
                    "residual": {"x": dx, "y": dy, "manhattan": man},
                }
            )
        else:
            code = global_best.discarded.get(i, "outlier_excluded")
            reason = {
                "no_cell_within_tolerance": (
                    "在最优参数下没有任何格位在坐标容差内，判为杂点（划痕亮点）"
                ),
                "outlier_excluded": (
                    "为获得互异格位的一致对准并最小化弃点数，该点作为杂点弃去"
                ),
            }[code]
            discarded.append(
                {
                    "marker_id": p.id,
                    "position": [p.x, p.y],
                    "reason_code": code,
                    "reason": reason,
                }
            )

    assignments.sort(key=lambda a: a["marker_id"])
    discarded.sort(key=lambda d: d["marker_id"])

    return {
        "feasible": True,
        "grid": {"rows": rows, "cols": cols},
        "parameters": {
            "origin": [ox, oy],
            "basis_row": [b1x, b1y],
            "basis_col": [b2x, b2y],
            "determinant": det,
        },
        "assignments": assignments,
        "discarded": discarded,
        "used_marker_count": len(assignments),
        "discarded_count": len(discarded),
        "max_manhattan_residual": max_manhattan,
        "total_residual": total,
        "objective": [global_best.key[0], max_manhattan, total],
        "stats": stats,
    }
