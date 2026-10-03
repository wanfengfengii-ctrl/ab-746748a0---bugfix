#!/usr/bin/env python3
"""端到端冒烟：针对运行中的服务提交含漏读与杂点的复原请求并校验结果。

用法: BASE_URL=http://localhost:8080 python scripts/smoke.py
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8080").rstrip("/")

PASS, FAIL = 0, 0


def post(path, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BASE_URL + path, data=data, headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")


def scenario_missing_and_outliers():
    print("场景 1：漏读 + 两个杂点（4 行 3 列，行基 [3,1]，列基 [-2,4]，det=14）")
    true = dict(ox=5, oy=-4, b1=(3, 1), b2=(-2, 4))
    real = [(0, 0), (0, 2), (1, 1), (2, 0), (2, 2), (3, 1), (3, 2)]
    markers = []
    for k, (r, c) in enumerate(real):
        markers.append({
            "id": k + 1,
            "x": true["ox"] + r * true["b1"][0] + c * true["b2"][0],
            "y": true["oy"] + r * true["b1"][1] + c * true["b2"][1],
        })
    markers += [
        {"id": 80, "x": 1000, "y": -1000},
        {"id": 81, "x": -777, "y": 777},
    ]
    payload = {
        "markers": markers, "rows": 4, "cols": 3, "tolerance": 0,
        "max_outliers": 2,
        "origin_x": {"lo": 2, "hi": 8}, "origin_y": {"lo": -7, "hi": -1},
        "basis_row_x": {"lo": 0, "hi": 6}, "basis_row_y": {"lo": -2, "hi": 4},
        "basis_col_x": {"lo": -5, "hi": 1}, "basis_col_y": {"lo": 1, "hi": 7},
    }
    status, body = post("/api/wafer-grids/reconstruct", payload)
    check("HTTP 200", status == 200, f"status={status} body={body}")
    check("feasible=true", body.get("feasible") is True)
    if not body.get("feasible"):
        return
    p = body["parameters"]
    check("原点恢复正确", p["origin"] == [true["ox"], true["oy"]], str(p["origin"]))
    check("行基向量恢复正确", p["basis_row"] == list(true["b1"]))
    check("列基向量恢复正确", p["basis_col"] == list(true["b2"]))
    check("行列式为正", p["determinant"] > 0 and p["determinant"] == 14)

    discarded = {d["marker_id"]: d for d in body["discarded"]}
    check("恰好弃去两个亮点", set(discarded) == {80, 81}, str(discarded))
    check("保留 7 个标记", body["used_marker_count"] == 7)
    check("弃点证据含原因", all(d["reason"] for d in discarded.values()))

    cells = [tuple(a["grid_cell"]) for a in body["assignments"]]
    check("格位互异", len(cells) == len(set(cells)), str(cells))
    check("格位数量正确", len(cells) == len(real))
    check("最大残差为 0", body["max_manhattan_residual"] == 0)
    check("残差总和为 0", body["total_residual"] == 0)
    expected_cell = {
        k + 1: cell for k, cell in enumerate(real)
    }
    ok = all(
        tuple(a["grid_cell"]) == expected_cell[a["marker_id"]]
        for a in body["assignments"]
    )
    check("逐点格位编号正确", ok)
    # 预测坐标自洽
    consistent = all(
        a["predicted"][0] == p["origin"][0]
        + a["grid_cell"][0] * p["basis_row"][0]
        + a["grid_cell"][1] * p["basis_col"][0]
        and a["predicted"][1] == p["origin"][1]
        + a["grid_cell"][0] * p["basis_row"][1]
        + a["grid_cell"][1] * p["basis_col"][1]
        for a in body["assignments"]
    )
    check("逐点预测坐标自洽", consistent)


def scenario_only_missing():
    print("场景 2：仅漏读无杂点（3x3 用 8 点，容差 1）")
    true = dict(ox=0, oy=0, b1=(4, 0), b2=(0, 4))
    real = [(r, c) for r in range(3) for c in range(3) if (r, c) != (1, 1)]
    jitter = {2: (1, 0), 5: (0, -1)}
    markers = []
    for k, (r, c) in enumerate(real):
        jx, jy = jitter.get(k + 1, (0, 0))
        markers.append({
            "id": k + 1,
            "x": r * 4 + jx, "y": c * 4 + jy,
        })
    payload = {
        "markers": markers, "rows": 3, "cols": 3, "tolerance": 1,
        "max_outliers": 2,
        "origin_x": {"lo": -2, "hi": 2}, "origin_y": {"lo": -2, "hi": 2},
        "basis_row_x": {"lo": 1, "hi": 6}, "basis_row_y": {"lo": -2, "hi": 2},
        "basis_col_x": {"lo": -2, "hi": 2}, "basis_col_y": {"lo": 1, "hi": 6},
    }
    status, body = post("/api/wafer-grids/reconstruct", payload)
    check("HTTP 200", status == 200, f"body={body}")
    check("feasible=true", body.get("feasible") is True)
    if body.get("feasible"):
        check("不弃点", body["discarded_count"] == 0)
        check("行列式为正", body["parameters"]["determinant"] > 0)
        cells = [tuple(a["grid_cell"]) for a in body["assignments"]]
        check("格位互异", len(cells) == len(set(cells)))
        check("残差不越界", all(
            abs(a["residual"]["x"]) <= 1 and abs(a["residual"]["y"]) <= 1
            for a in body["assignments"]))


def scenario_no_solution():
    print("场景 3：三个杂点但上限两个 -> 明确无解原因")
    markers = [
        {"id": 1, "x": 0, "y": 0}, {"id": 2, "x": 5, "y": 0},
        {"id": 3, "x": 10, "y": 0}, {"id": 4, "x": 0, "y": 5},
        {"id": 5, "x": 5, "y": 5}, {"id": 6, "x": 10, "y": 5},
        {"id": 7, "x": 0, "y": 10},
        {"id": 8, "x": 900, "y": 900},
        {"id": 9, "x": -900, "y": -900},
        {"id": 10, "x": 500, "y": -500},
    ]
    payload = {
        "markers": markers, "rows": 3, "cols": 3, "tolerance": 0,
        "max_outliers": 2,
        "origin_x": {"lo": -3, "hi": 3}, "origin_y": {"lo": -3, "hi": 3},
        "basis_row_x": {"lo": 2, "hi": 8}, "basis_row_y": {"lo": -2, "hi": 2},
        "basis_col_x": {"lo": -2, "hi": 2}, "basis_col_y": {"lo": 2, "hi": 8},
    }
    status, body = post("/api/wafer-grids/reconstruct", payload)
    check("HTTP 200", status == 200)
    check("feasible=false", body.get("feasible") is False)
    check("返回无解原因", bool(body.get("reasons")))
    if body.get("reasons"):
        check("原因可读且含“无解”",
              all("无解" in r["message"] for r in body["reasons"]),
              str(body["reasons"]))
        check("指出超预算标记",
              any(r.get("marker_ids") for r in body["reasons"]))


def scenario_bad_interval():
    print("场景 4：输入矛盾（区间跨度 7）-> 422 校验错误")
    payload = {
        "markers": [{"id": i, "x": i, "y": 0} for i in range(1, 8)],
        "rows": 3, "cols": 3, "tolerance": 0, "max_outliers": 2,
        "origin_x": {"lo": 0, "hi": 6}, "origin_y": {"lo": 0, "hi": 6},
        "basis_row_x": {"lo": 0, "hi": 7},
        "basis_row_y": {"lo": 0, "hi": 6},
        "basis_col_x": {"lo": 0, "hi": 6}, "basis_col_y": {"lo": 0, "hi": 6},
    }
    status, body = post("/api/wafer-grids/reconstruct", payload)
    check("HTTP 422", status == 422, f"status={status}")
    check("错误信息提及跨度", "跨度" in json.dumps(body, ensure_ascii=False))


def scenario_duplicate_ids():
    print("场景 5：标记编号重复 -> 422")
    payload = {
        "markers": [{"id": 1, "x": i, "y": 0} for i in range(7)],
        "rows": 3, "cols": 3, "tolerance": 0, "max_outliers": 2,
        "origin_x": {"lo": 0, "hi": 6}, "origin_y": {"lo": 0, "hi": 6},
        "basis_row_x": {"lo": 0, "hi": 6}, "basis_row_y": {"lo": 0, "hi": 6},
        "basis_col_x": {"lo": 0, "hi": 6}, "basis_col_y": {"lo": 0, "hi": 6},
    }
    status, body = post("/api/wafer-grids/reconstruct", payload)
    check("HTTP 422", status == 422)
    check("错误信息提及唯一编号", "唯一" in json.dumps(body, ensure_ascii=False))


def scenario_dense_identical_positions():
    print("场景 6：14 个全同坐标标记（7x7、容差 6、参数固定、不弃点）稠密回归")
    payload = {
        "markers": [{"id": i, "x": 0, "y": 0} for i in range(1, 15)],
        "rows": 7, "cols": 7, "tolerance": 6, "max_outliers": 0,
        "origin_x": {"lo": 0, "hi": 0}, "origin_y": {"lo": 0, "hi": 0},
        "basis_row_x": {"lo": 1, "hi": 1},
        "basis_row_y": {"lo": 0, "hi": 0},
        "basis_col_x": {"lo": 0, "hi": 0},
        "basis_col_y": {"lo": 1, "hi": 1},
    }
    import time
    t0 = time.monotonic()
    status, body = post("/api/wafer-grids/reconstruct", payload)
    elapsed = time.monotonic() - t0
    check("HTTP 200", status == 200, f"status={status} body={body}")
    check("feasible=true", body.get("feasible") is True)
    check("及时返回（< 8s）", elapsed < 8, f"耗时 {elapsed:.2f}s")
    if not body.get("feasible"):
        return
    check("不弃点", body["discarded_count"] == 0)
    check("保留 14 个标记", body["used_marker_count"] == 14)
    check("最大曼哈顿残差为 4", body["max_manhattan_residual"] == 4)
    check("残差总和为 36", body["total_residual"] == 36)
    expected = (
        [(0, c) for c in range(5)]
        + [(1, c) for c in range(4)]
        + [(2, c) for c in range(3)]
        + [(3, c) for c in range(2)]
    )
    got = [tuple(a["grid_cell"]) for a in body["assignments"]]
    check("格位序列为字典序最优", got == expected, str(got))
    check("14 个格位互异", len(set(got)) == 14)


def main():
    print(f"冒烟目标: {BASE_URL}")
    scenario_missing_and_outliers()
    scenario_only_missing()
    scenario_no_solution()
    scenario_bad_interval()
    scenario_duplicate_ids()
    scenario_dense_identical_positions()
    print(f"\n冒烟结果: {PASS} 通过, {FAIL} 失败")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
