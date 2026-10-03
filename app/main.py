"""晶圆标记栅格复原 HTTP 服务。"""
from __future__ import annotations

from fastapi import FastAPI

from .models import ReconstructRequest, ReconstructResponse
from .solver import solve

app = FastAPI(
    title="Wafer Grid Reconstruction Service",
    version="1.0.0",
    description="从含漏读与杂点的无序标记坐标中恢复整数栅格编号与对准参数。",
)


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/api/wafer-grids/reconstruct", response_model=ReconstructResponse)
def reconstruct(req: ReconstructRequest):
    result = solve(
        points=[{"id": m.id, "x": m.x, "y": m.y} for m in req.markers],
        rows=req.rows,
        cols=req.cols,
        tolerance=req.tolerance,
        origin_x=req.origin_x.as_tuple(),
        origin_y=req.origin_y.as_tuple(),
        basis_row_x=req.basis_row_x.as_tuple(),
        basis_row_y=req.basis_row_y.as_tuple(),
        basis_col_x=req.basis_col_x.as_tuple(),
        basis_col_y=req.basis_col_y.as_tuple(),
        max_outliers=req.max_outliers,
    )
    # 无解也算一次成功的“判定”，用 200 + feasible=false 明确给出无解原因；
    # 仅请求本身矛盾（参数越界等）走 422。
    return ReconstructResponse(**result)
