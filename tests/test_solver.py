"""求解器单元测试：覆盖漏读、杂点、容差、互异格位与最优性。"""
from __future__ import annotations

import itertools
import random
import time

import pytest

from app.solver import solve

TRUE = dict(ox=10, oy=-5, b1x=5, b1y=1, b2x=-2, b2y=4)  # det = 22 > 0


def grid_point(r, c, jitter=(0, 0)):
    x = TRUE["ox"] + r * TRUE["b1x"] + c * TRUE["b2x"] + jitter[0]
    y = TRUE["oy"] + r * TRUE["b1y"] + c * TRUE["b2y"] + jitter[1]
    return x, y


def tight_intervals(span=2):
    t = TRUE
    return dict(
        origin_x=(t["ox"] - span, t["ox"] + span),
        origin_y=(t["oy"] - span, t["oy"] + span),
        basis_row_x=(t["b1x"] - span, t["b1x"] + span),
        basis_row_y=(t["b1y"] - span, t["b1y"] + span),
        basis_col_x=(t["b2x"] - span, t["b2x"] + span),
        basis_col_y=(t["b2y"] - span, t["b2y"] + span),
    )


def make_markers(cells, start_id=1, jitter_seed=None):
    rng = random.Random(jitter_seed)
    markers = []
    for k, (r, c) in enumerate(cells):
        jx = jy = 0
        if jitter_seed is not None:
            jx, jy = rng.choice([(-1, 0), (1, 1), (0, -1), (0, 0)])
        x, y = grid_point(r, c, (jx, jy))
        markers.append({"id": start_id + k, "x": x, "y": y})
    return markers


def assert_true_params(res):
    p = res["parameters"]
    assert p["origin"] == [TRUE["ox"], TRUE["oy"]]
    assert p["basis_row"] == [TRUE["b1x"], TRUE["b1y"]]
    assert p["basis_col"] == [TRUE["b2x"], TRUE["b2y"]]
    assert p["determinant"] == TRUE["b1x"] * TRUE["b2y"] - TRUE["b1y"] * TRUE["b2x"]


def assert_distinct_cells(res):
    cells = [tuple(a["grid_cell"]) for a in res["assignments"]]
    assert len(cells) == len(set(cells)), "两个标记占用了同一格位"
    for a in res["assignments"]:
        r, c = a["grid_cell"]
        assert 0 <= r < res["grid"]["rows"]
        assert 0 <= c < res["grid"]["cols"]
        assert abs(a["residual"]["x"]) <= TOL
        assert abs(a["residual"]["y"]) <= TOL
        assert a["residual"]["manhattan"] == abs(a["residual"]["x"]) + abs(
            a["residual"]["y"]
        )
        p = res["parameters"]
        px = p["origin"][0] + r * p["basis_row"][0] + c * p["basis_col"][0]
        py = p["origin"][1] + r * p["basis_row"][1] + c * p["basis_col"][1]
        assert a["predicted"] == [px, py]


TOL = 1


def test_perfect_grid_recovery_scrambled():
    # 3x3 中 7 个真实标记（2 个漏读），乱序提交
    cells = [(0, 0), (0, 2), (1, 1), (2, 0), (2, 1), (2, 2), (0, 1)]
    markers = make_markers(cells)
    rng = random.Random(42)
    rng.shuffle(markers)
    res = solve(markers, 3, 3, 0, max_outliers=2, **tight_intervals())
    assert res["feasible"]
    assert_true_params(res)
    assert res["discarded_count"] == 0
    assert res["used_marker_count"] == 7
    assert res["max_manhattan_residual"] == 0
    assert res["total_residual"] == 0
    assert_distinct_cells(res)
    expected = {1 + k: cell for k, cell in enumerate(cells)}
    cell_of = {a["marker_id"]: tuple(a["grid_cell"]) for a in res["assignments"]}
    for mid, cell in expected.items():
        assert cell_of[mid] == cell


def test_missing_and_two_outliers():
    # 4x4 上 8 个真实标记（8 个漏读）+ 2 个划痕亮点
    real_cells = [(0, 0), (0, 3), (1, 1), (1, 2), (2, 0), (2, 3), (3, 1), (3, 3)]
    markers = make_markers(real_cells)
    markers.append({"id": 90, "x": 200, "y": 200})   # 远处亮点
    markers.append({"id": 91, "x": -150, "y": 99})
    rng = random.Random(7)
    rng.shuffle(markers)
    res = solve(markers, 4, 4, 0, max_outliers=2, **tight_intervals())
    assert res["feasible"]
    assert_true_params(res)
    assert {d["marker_id"] for d in res["discarded"]} == {90, 91}
    assert res["discarded_count"] == 2
    assert res["used_marker_count"] == 8
    assert_distinct_cells(res)
    for d in res["discarded"]:
        assert d["reason_code"] == "no_cell_within_tolerance"
        assert d["position"]


def test_tolerance_jitter():
    cells = [(0, 0), (1, 2), (2, 1), (0, 1), (1, 0), (2, 2), (1, 1)]
    markers = make_markers(cells, jitter_seed=123)
    res = solve(markers, 3, 3, TOL, max_outliers=2, **tight_intervals())
    assert res["feasible"]
    # 含抖动时平移原点可得到同残差的等价解，按参数字典序择优；
    # 这里校验一致性而非唯一真参数
    p = res["parameters"]
    assert p["determinant"] > 0
    assert res["discarded_count"] == 0
    assert res["max_manhattan_residual"] <= 2 * TOL
    assert res["total_residual"] == sum(
        a["residual"]["manhattan"] for a in res["assignments"]
    )
    assert res["max_manhattan_residual"] == max(
        a["residual"]["manhattan"] for a in res["assignments"]
    )
    assert_distinct_cells(res)


def test_three_outliers_infeasible():
    cells = [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)]
    markers = make_markers(cells)
    markers += [
        {"id": 80, "x": 300, "y": 300},
        {"id": 81, "x": -300, "y": -300},
        {"id": 82, "x": 0, "y": 400},
    ]
    res = solve(markers, 3, 3, 0, max_outliers=2, **tight_intervals())
    assert not res["feasible"]
    codes = [r["code"] for r in res["reasons"]]
    assert "too_many_unmatchable_markers" in codes
    reason = next(r for r in res["reasons"] if r["code"] == "too_many_unmatchable_markers")
    assert set(reason["marker_ids"]) == {80, 81, 82}
    assert "无解" in reason["message"]


def test_duplicate_position_cell_conflict_forces_one_outlier():
    # 两个标记落在同一物理位置：不能占同一格位，必须弃一
    markers = make_markers([(0, 0), (0, 2), (1, 1), (2, 0), (2, 2), (1, 0)])
    dup = dict(markers[0])
    dup["id"] = 77
    markers.append(dup)
    res = solve(markers, 3, 3, 0, max_outliers=2, **tight_intervals())
    assert res["feasible"]
    assert res["discarded_count"] == 1
    # 同位置的两个标记中恰有一个被弃（字典序决定具体哪个）
    discarded_id = res["discarded"][0]["marker_id"]
    pair = {markers[0]["id"], 77}
    assert discarded_id in pair
    kept_id = next(iter(pair - {discarded_id}))
    kept = next(a for a in res["assignments"] if a["marker_id"] == kept_id)
    assert tuple(kept["grid_cell"]) == (0, 0)
    assert_distinct_cells(res)


def test_determinant_must_be_positive():
    # b1 固定为行方向，b2 固定为翻转后的列方向：det = 5*(-4) - 1*(-2) = -18
    iv = tight_intervals()
    iv["basis_row_x"] = (TRUE["b1x"], TRUE["b1x"])
    iv["basis_row_y"] = (TRUE["b1y"], TRUE["b1y"])
    iv["basis_col_x"] = (TRUE["b2x"], TRUE["b2x"])
    iv["basis_col_y"] = (-TRUE["b2y"], -TRUE["b2y"])
    markers = make_markers([(0, 0), (0, 2), (1, 1), (2, 0), (2, 2), (1, 0), (2, 1)])
    res = solve(markers, 3, 3, 0, max_outliers=2, **iv)
    assert not res["feasible"]
    assert res["reasons"][0]["code"] == "no_positive_determinant_basis"


def test_deterministic_across_runs():
    cells = [(r, c) for r in range(3) for c in range(2)] + [(2, 2)]
    markers = make_markers(cells, jitter_seed=9)
    markers.append({"id": 55, "x": 123, "y": -77})
    kw = dict(rows=3, cols=3, tolerance=TOL, max_outliers=2, **tight_intervals(3))
    r1 = solve(markers, **kw)
    r2 = solve(markers, **kw)
    assert r1 == r2


def test_min_outliers_priority_over_residual():
    # 存在“弃 1 点残差更小”的诱惑时，仍须选择 0 弃点
    cells = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 1), (2, 2), (0, 2)]
    markers = make_markers(cells, jitter_seed=5)
    res = solve(markers, 3, 3, TOL, max_outliers=2, **tight_intervals())
    assert res["feasible"]
    assert res["discarded_count"] == 0
    assert_distinct_cells(res)


def test_bruteforce_optimality_small():
    """独立暴力枚举全部参数与分配，验证字典序最优。"""
    cells = [(0, 0), (1, 1), (2, 0), (0, 2)]
    markers = make_markers(cells)
    markers.append({"id": 40, "x": 50, "y": 60})  # 1 个杂点
    iv = tight_intervals(1)
    res = solve(markers, 3, 3, TOL, max_outliers=2, **iv)
    assert res["feasible"]

    pts = sorted(((m["x"], m["y"]) for m in markers))
    best_key = None
    ranges = [iv[k] for k in (
        "origin_x", "origin_y", "basis_row_x", "basis_row_y",
        "basis_col_x", "basis_col_y")]
    all_cells = [(r, c) for r in range(3) for c in range(3)]
    for ox, oy, b1x, b1y, b2x, b2y in itertools.product(
        *(range(a, b + 1) for a, b in ranges)
    ):
        if b1x * b2y - b1y * b2x <= 0:
            continue
        cand = []
        for x, y in pts:
            opts = []
            for r, c in all_cells:
                dx = x - (ox + r * b1x + c * b2x)
                dy = y - (oy + r * b1y + c * b2y)
                if abs(dx) <= TOL and abs(dy) <= TOL:
                    opts.append(((r, c), abs(dx) + abs(dy)))
            cand.append(opts)

        def enum(i, used, d, mx, tot, seq):
            nonlocal best_key
            if d > 2:
                return
            if i == len(pts):
                key = (d, mx, tot, (ox, oy, b1x, b1y, b2x, b2y), tuple(seq))
                if best_key is None or key < best_key:
                    best_key = key
                return
            for cell, man in cand[i]:
                if cell not in used:
                    enum(i + 1, used | {cell}, d, max(mx, man), tot + man,
                         seq + [cell])
            enum(i + 1, used, d + 1, mx, tot, seq + [(-1, -1)])

        enum(0, set(), 0, 0, 0, [])

    assert res["objective"][:3] == list(best_key[:3])
    p = res["parameters"]
    assert tuple(p["origin"] + p["basis_row"] + p["basis_col"]) == best_key[3]
    assert res["discarded_count"] == best_key[0]


def test_worst_case_runtime():
    # 7x7 栅格、14 个点、所有区间跨度 6：性能回归保护
    rng = random.Random(2024)
    cells = rng.sample([(r, c) for r in range(7) for c in range(7)], 12)
    markers = make_markers(cells, jitter_seed=11)
    # 所有六条区间跨度均取上限 6，且包含真实参数
    wide = dict(
        origin_x=(7, 13), origin_y=(-8, -2),
        basis_row_x=(2, 8), basis_row_y=(-2, 4),
        basis_col_x=(-5, 1), basis_col_y=(1, 7),
    )
    t0 = time.monotonic()
    res = solve(markers, 7, 7, TOL, max_outliers=2, **wide)
    elapsed = time.monotonic() - t0
    assert res["feasible"]
    assert res["discarded_count"] <= 2
    assert_distinct_cells(res)
    assert elapsed < 30, f"最坏情形耗时 {elapsed:.1f}s，超过 30s 预算"
