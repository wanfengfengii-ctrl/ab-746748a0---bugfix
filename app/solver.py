"""精确求解器：从可能含漏读（空格位）与杂点（划痕亮点）的标记坐标中恢复整数栅格。

联合优化：
  1. 原点 o、行基向量 b1、列基向量 b2（均在给定闭区间内，且 det(b1,b2) > 0）；
  2. 每个保留标记到栅格内互异格位 (r, c) 的分配（任意两个标记不得占用同一格位）；
  3. 弃点（杂点）数量不超过 max_outliers。

目标按字典序最小化：
  (弃点数, 最大曼哈顿残差, 残差总和, 完整参数序列, 完整分配序列)

搜索方式：六条闭区间跨度均不超过 6，基向量组合至多 7^4 个，先按行列式为正过滤，
再利用“标记必须能由 原点 = 坐标 - r*b1 - c*b2 落在原点区间”做快速预筛；对每组
参数，候选格位按分量容差预筛后，互异分配用多项式时间的精确算法求解，避免在
稠密同坐标（所有标记候选格位完全相同）场景下枚举指数级的对称排列：

  1. 二分图最大匹配（Kuhn + 位集）求最少弃点数 d；
  2. 对残差阈值二分 + 最大匹配，求可行的最小最大曼哈顿残差 T；
  3. 补 d 个零费用“弃点”列做矩形指派（匈牙利算法），求最小残差总和 S；
  4. 按标记编号逐位贪心，以“剩余问题的最小费用指派能否恰好补齐 S”为预言，
     构造字典序最小的完整分配序列（弃点记为 (-1,-1)，是该位置的最小取值）。

全部为整数运算，结果确定、可复现。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

Cell = tuple[int, int]
DISCARD_CELL = (-1, -1)

# 残差上界：|dx|+|dy| ≤ 2*tolerance ≤ 12，总和 ≤ 14*12 = 168；
# 取足够大的 INF 表示“该标记不能使用此格位/弃点名额”。
_INF = 10**9


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


def _kuhn_size(adj, n_cells):
    """二分图最大匹配（Kuhn 算法）。adj[i] 为标记 i 邻接格位压缩编号的位集。

    返回最大匹配边数。稠密同坐标场景下所有标记邻接相同的格位集合，
    最大匹配边数可直接取 min(标记数, 格位数)，免去递归增广。
    """
    if adj:
        first = 0
        for bits in adj:
            if bits:
                first = bits
                break
        if all(bits == 0 or bits == first for bits in adj):
            active = sum(1 for bits in adj if bits)
            return min(active, first.bit_count())
    owner = [-1] * n_cells
    seen = [0] * n_cells
    stamp = 0

    def augment(mi):
        bits = adj[mi]
        while bits:
            lb = bits & -bits
            j = lb.bit_length() - 1
            bits ^= lb
            if seen[j] == stamp:
                continue
            seen[j] = stamp
            if owner[j] < 0 or augment(owner[j]):
                owner[j] = mi
                return True
        return False

    size = 0
    for mi in range(len(adj)):
        stamp += 1
        if augment(mi):
            size += 1
    return size


def _hungarian(cost):
    """矩形指派问题（行数 n ≤ 列数 m）的匈牙利算法，返回 (最小费用, 行→列指派)。

    费用矩阵中 >= _INF 的边表示不可行；若最优指派仍被迫使用不可行边则返回 None。
    """
    n = len(cost)
    m = len(cost[0])
    u = [0] * (n + 1)
    v = [0] * (m + 1)
    p = [0] * (m + 1)   # p[j]：占用第 j 列的行（1 基），0 表示未占用
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [_INF] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = _INF
            j1 = 0
            row = cost[i0 - 1]
            ui = u[i0]
            for j in range(1, m + 1):
                if not used[j]:
                    cur = row[j - 1] - ui - v[j]
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

    assign = [-1] * n
    total = 0
    for j in range(1, m + 1):
        i = p[j]
        if i:
            c = cost[i - 1][j - 1]
            if c >= _INF:
                return None
            total += c
            assign[i - 1] = j - 1
    return total, assign


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

    # 按物理坐标去重：同一坐标的标记在任意一组参数下候选格位完全一致，
    # 整组共享同一份候选列表，稠密同坐标场景下把重复计算降到一次。
    pos_groups: dict[tuple[int, int], list[int]] = {}
    for i, p in enumerate(pts):
        pos_groups.setdefault((p.x, p.y), []).append(i)
    uniq_pos = list(pos_groups.keys())

    ever_fit = [False] * n          # 每个标记是否曾在某参数下落入某格位
    any_all_fit = False             # 是否存在所有标记单独都能落入格位的参数
    # 全局最优只按字典序前四项（弃点数、最大残差、残差和、参数序列）裁决；
    # 分配序列是末位优先级，只需对唯一胜出的参数集构造一次。
    best_head: Optional[tuple] = None
    winner: Optional[dict] = None

    def min_assignment(edges_le, cell_idx, n_cells, remain, used_cells,
                       discards_left):
        """剩余标记在已占格位之外、恰有 discards_left 个弃点名额时的最小指派。

        费用矩阵列 = 未占候选格位（费用为曼哈顿残差，无边为 _INF）+ 每列容量 1、
        费用 0 的弃点列；返回 (最小残差和, 行→列指派, 真实格位列数)，不可行返回 None。
        """
        m = len(remain)
        if discards_left < 0:
            return None
        free = [j for j in range(n_cells) if j not in used_cells]
        nf = len(free)
        if m == 0:
            return (0, [], nf) if discards_left == 0 else None
        free_pos = {j: k for k, j in enumerate(free)}
        ncols = nf + discards_left
        if m > ncols:
            return None
        cost = [[_INF] * nf + [0] * discards_left for _ in range(m)]
        for k, mi in enumerate(remain):
            for r, c, res in edges_le[mi]:
                q = free_pos.get(cell_idx[(r, c)])
                if q is not None and res < cost[k][q]:
                    cost[k][q] = res
        hung = _hungarian(cost)
        if hung is None:
            return None
        total, row_assign = hung
        if sum(1 for a in row_assign if a >= nf) != discards_left:
            return None
        return total, row_assign, nf

    def consider(ox, oy, b1x, b1y, b2x, b2y, cand, forced):
        """评估一组参数：求 (最少弃点数 d, 最小最大残差 T, 最小残差和 S) 并裁汰。"""
        nonlocal best_head, winner, any_all_fit
        stats["parameter_sets_evaluated"] += 1
        if not forced:
            any_all_fit = True
        for i in range(n):
            if cand[i]:
                ever_fit[i] = True
        if len(forced) > budget:
            return

        # 压缩候选格位编号并构造邻接位集
        cell_idx: dict[Cell, int] = {}
        for opts in cand:
            for r, c, _res in opts:
                if (r, c) not in cell_idx:
                    cell_idx[(r, c)] = len(cell_idx)
        n_cells = len(cell_idx)
        adj = [0] * n
        for i, opts in enumerate(cand):
            bits = 0
            for r, c, _res in opts:
                bits |= 1 << cell_idx[(r, c)]
            adj[i] = bits

        # 1) 最少弃点数 d = n - 最大匹配
        matched = _kuhn_size(adj, n_cells)
        d = n - matched
        if d > budget:
            return

        # 2) 最小最大残差 T：对候选残差阈值二分，阈值图仍须容下 matched 个匹配
        residuals = sorted({res for opts in cand for _, _, res in opts})
        lo_t, hi_t = 0, len(residuals) - 1
        while lo_t < hi_t:
            mid = (lo_t + hi_t) // 2
            t = residuals[mid]
            sub = [0] * n
            for i, opts in enumerate(cand):
                bits = 0
                for r, c, res in opts:
                    if res <= t:
                        bits |= 1 << cell_idx[(r, c)]
                sub[i] = bits
            if _kuhn_size(sub, n_cells) >= matched:
                hi_t = mid
            else:
                lo_t = mid + 1
        T = residuals[lo_t]
        if best_head is not None:
            if d > best_head[0] or (d == best_head[0] and T > best_head[1]):
                return

        edges_le = [
            [(r, c, res) for r, c, res in opts if res <= T]
            for opts in cand
        ]

        # 3) 最小残差和 S：先用两个松弛下界快速裁汰，再做矩形指派。
        #    LB1：忽略格位互斥，各标记独立取最低残差（d 个弃点名额用于豁免
        #    最贵的 d 个）；LB2：忽略标记-格位归属，每个格位取所有可达标记的
        #    最低残差，取最便宜的 n-d 个真实格位。二者均为 S 的合法下界，
        #    在稠密同坐标（各标记候选完全相同）场景下 LB2 恰好等于 S。
        non_forced = [i for i in range(n) if edges_le[i]]
        marker_mins = sorted(min(res for _, _, res in edges_le[i])
                             for i in non_forced)
        lb1 = sum(marker_mins[: n - d])
        cell_min: dict[int, int] = {}
        for opts in edges_le:
            for r, c, res in opts:
                j = cell_idx[(r, c)]
                if j not in cell_min or res < cell_min[j]:
                    cell_min[j] = res
        lb2 = sum(sorted(cell_min.values())[: n - d])
        lb = max(lb1, lb2)
        params_key = (ox, oy, b1x, b1y, b2x, b2y)
        if best_head is not None and d == best_head[0] and T == best_head[1]:
            if lb > best_head[2]:
                return
        # d 或 T 严格优于当前最优时，下界不可能裁汰，必须计算精确的 S

        full = min_assignment(edges_le, cell_idx, n_cells,
                              list(range(n)), frozenset(), d)
        if full is None:
            return
        S = full[0]
        head = (d, T, S, params_key)
        if best_head is not None and head >= best_head:
            return
        best_head = head
        winner = {
            "forced": forced,
            "cell_idx": cell_idx,
            "n_cells": n_cells,
            "edges_le": edges_le,
            "d": d,
            "T": T,
            "S": S,
        }


    for b1x, b1y, b2x, b2y in basis_pairs:
        # centers_u[g] = 该唯一坐标组的 [(r, c, cx, cy)]，cx/cy 使标记落在
        # (r,c) 时原点应为 (cx, cy) = p - r*b1 - c*b2；同坐标组共享一份
        centers_u = []
        feasible_markers = 0
        group_fits = []
        for gx, gy in uniq_pos:
            lst = []
            feasible = False
            for r, c in cells:
                cx = gx - r * b1x - c * b2x
                cy = gy - r * b1y - c * b2y
                lst.append((r, c, cx, cy))
                # 存在原点区间内（含容差扩张）的 o 使该格位容纳此标记
                if (
                    ox_lo - tolerance <= cx <= ox_hi + tolerance
                    and oy_lo - tolerance <= cy <= oy_hi + tolerance
                ):
                    feasible = True
            centers_u.append(lst)
            group_fits.append(feasible)
            if feasible:
                feasible_markers += len(pos_groups[(gx, gy)])

        if feasible_markers < need:
            stats["basis_pairs_rejected_pre_filter"] += 1
            for g, (gx, gy) in enumerate(uniq_pos):
                if group_fits[g]:
                    for i in pos_groups[(gx, gy)]:
                        ever_fit[i] = True
            continue
        for g, (gx, gy) in enumerate(uniq_pos):
            if group_fits[g]:
                for i in pos_groups[(gx, gy)]:
                    ever_fit[i] = True

        for ox in range(ox_lo, ox_hi + 1):
            for oy in range(oy_lo, oy_hi + 1):
                # 每个唯一物理坐标只计算一次候选格位列表，组内标记共享
                opts_by_group = []
                forced_groups = set()
                for g, lst in enumerate(centers_u):
                    gx, gy = uniq_pos[g]
                    opts = []
                    for r, c, _cx, _cy in lst:
                        dx = gx - (ox + r * b1x + c * b2x)
                        dy = gy - (oy + r * b1y + c * b2y)
                        if abs(dx) <= tolerance and abs(dy) <= tolerance:
                            opts.append((r, c, abs(dx) + abs(dy)))
                    if opts:
                        opts.sort(key=lambda t: (t[2], t[0], t[1]))
                    else:
                        forced_groups.add(g)
                    opts_by_group.append(opts)
                cand = [None] * n
                forced = set()
                for g, idxs in enumerate(pos_groups.values()):
                    for i in idxs:
                        cand[i] = opts_by_group[g]
                        if g in forced_groups:
                            forced.add(i)
                consider(ox, oy, b1x, b1y, b2x, b2y, cand, forced)

    if winner is None:
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

    # 对唯一胜出的参数集构造字典序最小的完整分配序列（末位优先级，只做一次）
    d = winner["d"]
    T = winner["T"]
    S = winner["S"]
    params_key = best_head[3]
    edges_le = winner["edges_le"]
    cell_idx = winner["cell_idx"]
    n_cells = winner["n_cells"]
    forced = winner["forced"]

    def feasible_prefix(pos, used_cells, discards_used, cost_so_far):
        """pos 之前的标记已定案，剩余位置能否恰弃 d 个点且总费用补齐到 S。"""
        stats["complete_assignments_explored"] += 1
        rem = min_assignment(edges_le, cell_idx, n_cells,
                             list(range(pos, n)), used_cells, d - discards_used)
        return rem is not None and rem[0] == S - cost_so_far

    used_cells: set[int] = set()
    assign: dict[int, Cell] = {}
    discarded: dict[int, str] = {}
    discards_used = 0
    cost_so_far = 0
    for i in range(n):
        if i in forced:
            # 无候选格位：只能弃去（d 已含此类点，可行性由整体计算保证）
            discarded[i] = "no_cell_within_tolerance"
            discards_used += 1
            continue
        # (-1,-1) 是该位置字典序最小的取值，优先尝试弃去
        if discards_used < d and feasible_prefix(
            i + 1, used_cells, discards_used + 1, cost_so_far
        ):
            discarded[i] = "outlier_excluded"
            discards_used += 1
            continue
        # 否则按 (r,c) 字典序尝试各格位，取第一个可补齐最优解的
        choice: Optional[tuple[Cell, int]] = None
        for r, c, res in sorted(edges_le[i], key=lambda t: (t[0], t[1])):
            j = cell_idx[(r, c)]
            if j in used_cells:
                continue
            used_cells.add(j)
            if feasible_prefix(i + 1, used_cells, discards_used, cost_so_far + res):
                choice = ((r, c), res)
                break
            used_cells.discard(j)
        if choice is None:
            # 格位均不成立则弃去；整体可行性已由整题指派保证此处必然可走
            discarded[i] = "outlier_excluded"
            discards_used += 1
        else:
            (r, c), res = choice
            used_cells.add(cell_idx[(r, c)])
            assign[i] = (r, c)
            cost_so_far += res

    seq = tuple(assign[i] if i in assign else DISCARD_CELL for i in range(n))
    global_best = _Solution(
        (d, T, S, params_key, seq), dict(assign), dict(discarded)
    )

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
