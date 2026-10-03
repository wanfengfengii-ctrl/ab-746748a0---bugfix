"""API 层测试：健康检查、成功复原、无解与输入校验。"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def base_payload(**overrides):
    payload = {
        "markers": [
            # 3x3 栅格原点 [0,0]，行基向量 [4,1]，列基向量 [-1,3]（det=13）
            # 含 2 个漏读，7 个真实标记 + 2 个亮点
            {"id": 1, "x": 0, "y": 0},
            {"id": 2, "x": -2, "y": 6},
            {"id": 3, "x": 4, "y": 1},
            {"id": 4, "x": 3, "y": 4},
            {"id": 5, "x": 8, "y": 2},
            {"id": 6, "x": 6, "y": 8},
            {"id": 7, "x": 7, "y": 5},
            {"id": 8, "x": 99, "y": 99},
            {"id": 9, "x": -88, "y": 40},
        ],
        "rows": 3,
        "cols": 3,
        "tolerance": 0,
        "max_outliers": 2,
        "origin_x": {"lo": -3, "hi": 3},
        "origin_y": {"lo": -3, "hi": 3},
        "basis_row_x": {"lo": 1, "hi": 7},
        "basis_row_y": {"lo": -2, "hi": 4},
        "basis_col_x": {"lo": -4, "hi": 2},
        "basis_col_y": {"lo": 0, "hi": 6},
    }
    payload.update(overrides)
    return payload


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_reconstruct_with_missing_and_outliers():
    r = client.post("/api/wafer-grids/reconstruct", json=base_payload())
    assert r.status_code == 200
    body = r.json()
    assert body["feasible"] is True
    assert body["parameters"]["origin"] == [0, 0]
    assert body["parameters"]["basis_row"] == [4, 1]
    assert body["parameters"]["basis_col"] == [-1, 3]
    assert body["parameters"]["determinant"] == 13
    assert {d["marker_id"] for d in body["discarded"]} == {8, 9}
    assert body["used_marker_count"] == 7
    assert body["discarded_count"] == 2
    # 互异格位
    cells = [tuple(a["grid_cell"]) for a in body["assignments"]]
    assert len(cells) == len(set(cells))
    # 每个标记的格位预测与残差自洽
    for a in body["assignments"]:
        assert abs(a["residual"]["x"]) == 0
        assert a["predicted"][:2] == [
            a["predicted"][0], a["predicted"][1],
        ]
    # 弃点证据完整
    for d in body["discarded"]:
        assert d["reason"]
        assert d["reason_code"] in {
            "no_cell_within_tolerance", "outlier_excluded"}


def test_reconstruct_no_solution_explicit_reasons():
    p = base_payload()
    p["max_outliers"] = 1  # 两个亮点无法只弃一个
    r = client.post("/api/wafer-grids/reconstruct", json=p)
    assert r.status_code == 200
    body = r.json()
    assert body["feasible"] is False
    assert body["parameters"] is None
    codes = {x["code"] for x in body["reasons"]}
    assert codes  # 至少一条明确原因
    for reason in body["reasons"]:
        assert reason["message"]
        assert "无解" in reason["message"]


def test_reject_duplicate_marker_ids():
    p = base_payload()
    p["markers"][-1]["id"] = 1
    r = client.post("/api/wafer-grids/reconstruct", json=p)
    assert r.status_code == 422
    assert "唯一" in r.text


def test_reject_bad_interval_span():
    p = base_payload()
    p["basis_row_x"] = {"lo": 0, "hi": 7}  # 跨度 7 > 6
    r = client.post("/api/wafer-grids/reconstruct", json=p)
    assert r.status_code == 422
    assert "跨度" in r.text


def test_reject_inverted_interval():
    p = base_payload()
    p["origin_y"] = {"lo": 5, "hi": 1}
    r = client.post("/api/wafer-grids/reconstruct", json=p)
    assert r.status_code == 422


def test_reject_marker_count_out_of_range():
    p = base_payload()
    p["markers"] = p["markers"][:5]  # 少于 7
    r = client.post("/api/wafer-grids/reconstruct", json=p)
    assert r.status_code == 422


def test_reject_bad_grid_dimensions():
    p = base_payload(rows=2, cols=3)
    r = client.post("/api/wafer-grids/reconstruct", json=p)
    assert r.status_code == 422
    p = base_payload(rows=8, cols=3)
    r = client.post("/api/wafer-grids/reconstruct", json=p)
    assert r.status_code == 422


def test_reject_negative_tolerance():
    p = base_payload(tolerance=-1)
    r = client.post("/api/wafer-grids/reconstruct", json=p)
    assert r.status_code == 422


def test_reject_too_many_outliers():
    p = base_payload(max_outliers=3)
    r = client.post("/api/wafer-grids/reconstruct", json=p)
    assert r.status_code == 422
