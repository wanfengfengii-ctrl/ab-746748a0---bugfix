"""请求/响应数据模型与输入校验。"""
from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field, model_validator

MAX_SPAN = 6
MIN_MARKERS, MAX_MARKERS = 7, 14
MIN_DIM, MAX_DIM = 3, 7
MAX_OUTLIERS_LIMIT = 2


class MarkerIn(BaseModel):
    id: int = Field(..., description="标记的唯一编号")
    x: int
    y: int


class IntervalIn(BaseModel):
    lo: int = Field(..., description="闭区间下界（含）")
    hi: int = Field(..., description="闭区间上界（含）")

    @model_validator(mode="after")
    def _check(self):
        if self.lo > self.hi:
            raise ValueError(f"闭区间下界 {self.lo} 不得大于上界 {self.hi}")
        if self.hi - self.lo > MAX_SPAN:
            raise ValueError(
                f"区间 [{self.lo}, {self.hi}] 跨度 {self.hi - self.lo} 超过 {MAX_SPAN}"
            )
        return self

    def as_tuple(self) -> tuple[int, int]:
        return (self.lo, self.hi)


class ReconstructRequest(BaseModel):
    markers: List[MarkerIn] = Field(
        ..., min_length=MIN_MARKERS, max_length=MAX_MARKERS
    )
    rows: int = Field(..., ge=MIN_DIM, le=MAX_DIM)
    cols: int = Field(..., ge=MIN_DIM, le=MAX_DIM)
    tolerance: int = Field(0, ge=0, description="坐标分量容差（整数，|Δx|、|Δy| 均不越过）")
    max_outliers: int = Field(2, ge=0, le=MAX_OUTLIERS_LIMIT)

    origin_x: IntervalIn
    origin_y: IntervalIn
    basis_row_x: IntervalIn
    basis_row_y: IntervalIn
    basis_col_x: IntervalIn
    basis_col_y: IntervalIn

    @model_validator(mode="after")
    def _check_markers(self):
        ids = [m.id for m in self.markers]
        if len(set(ids)) != len(ids):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"标记编号必须唯一，重复编号: {dupes}")
        return self


class Residual(BaseModel):
    x: int
    y: int
    manhattan: int


class AssignmentOut(BaseModel):
    marker_id: int
    grid_cell: List[int] = Field(..., description="[行号, 列号]，自 0 起")
    predicted: List[int] = Field(..., description="该格位预测坐标 [x, y]")
    residual: Residual


class DiscardEvidence(BaseModel):
    marker_id: int
    position: List[int]
    reason_code: Literal["no_cell_within_tolerance", "outlier_excluded"]
    reason: str


class ParametersOut(BaseModel):
    origin: List[int]
    basis_row: List[int]
    basis_col: List[int]
    determinant: int


class NoSolutionReason(BaseModel):
    code: str
    message: str
    marker_ids: List[int] | None = None


class ReconstructResponse(BaseModel):
    feasible: bool
    grid: dict | None = None
    parameters: ParametersOut | None = None
    assignments: List[AssignmentOut] = Field(default_factory=list)
    discarded: List[DiscardEvidence] = Field(default_factory=list)
    used_marker_count: int | None = None
    discarded_count: int | None = None
    max_manhattan_residual: int | None = None
    total_residual: int | None = None
    objective: List[int] | None = None
    reasons: List[NoSolutionReason] | None = None
    stats: dict | None = None
