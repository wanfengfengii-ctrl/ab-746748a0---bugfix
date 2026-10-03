# 晶圆标记栅格复原服务

从晶圆量测设备上报的无序整数坐标中恢复实际栅格编号与对准参数。数据中可能
存在**漏读**（栅格格位无标记）与**杂点**（划痕亮点），服务联合选择原点、
行/列基向量与标记到**互异格位**的分配，避免逐点吸附最近格位导致的对准参数冲突。

## 优化目标（字典序）

1. **弃点数最少**（不超过 `max_outliers`，最多 2）
2. **最大曼哈顿残差最小**
3. **残差总和最小**
4. 完整参数序列最小 `(origin, basis_row, basis_col)`
5. 完整分配序列最小（按标记编号排列的格位序列，弃点记为 `(-1,-1)`）

约束：

- 行基向量 b1、列基向量 b2 满足 `det(b1,b2) > 0`；
- 每个采用点的预测坐标逐分量满足 `|dx| ≤ tolerance`、`|dy| ≤ tolerance`；
- 任意两个标记不得占用同一格位。

## 请求

`POST /api/wafer-grids/reconstruct`

```json
{
  "markers": [
    {"id": 1, "x": 5, "y": -3}
  ],
  "rows": 4,
  "cols": 3,
  "tolerance": 1,
  "max_outliers": 2,
  "origin_x":     {"lo": 2, "hi": 8},
  "origin_y":     {"lo": -7, "hi": -1},
  "basis_row_x":  {"lo": 0, "hi": 6},
  "basis_row_y":  {"lo": -2, "hi": 4},
  "basis_col_x":  {"lo": -5, "hi": 1},
  "basis_col_y":  {"lo": 1, "hi": 7}
}
```

约束：7–14 个**唯一编号**标记；行列数 3–7；六条闭区间跨度（hi−lo）≤ 6；
`tolerance ≥ 0`；`max_outliers ∈ [0,2]`。输入矛盾返回 422 并说明具体字段。

## 响应

成功（`feasible: true`）返回参数、逐点格位与残差、弃点证据：

```json
{
  "feasible": true,
  "grid": {"rows": 4, "cols": 3},
  "parameters": {
    "origin": [5, -4],
    "basis_row": [3, 1],
    "basis_col": [-2, 4],
    "determinant": 14
  },
  "assignments": [
    {
      "marker_id": 1,
      "grid_cell": [0, 0],
      "predicted": [5, -4],
      "residual": {"x": 0, "y": 0, "manhattan": 0}
    }
  ],
  "discarded": [
    {
      "marker_id": 80,
      "position": [1000, -1000],
      "reason_code": "no_cell_within_tolerance",
      "reason": "在最优参数下没有任何格位在坐标容差内，判为杂点（划痕亮点）"
    }
  ],
  "used_marker_count": 7,
  "discarded_count": 2,
  "max_manhattan_residual": 0,
  "total_residual": 0,
  "objective": [0, 0, 0],
  "stats": {
    "positive_determinant_basis_pairs": 209,
    "basis_pairs_rejected_pre_filter": 0,
    "parameter_sets_evaluated": 300,
    "complete_assignments_explored": 1
  }
}
```

输入所允许的全部参数下均无解时（弃点超预算、格位冲突无法消解等），返回
`feasible: false` 并给出明确的无解原因（仍为 HTTP 200；仅请求格式/范围矛盾
才是 422）：

```json
{
  "feasible": false,
  "reasons": [
    {
      "code": "too_many_unmatchable_markers",
      "marker_ids": [8, 9, 10],
      "message": "标记 [8, 9, 10] 在所有允许的原点/基向量下都无法落入……故无解"
    }
  ],
  "stats": {"parameter_sets_evaluated": 0}
}
```

## 算法

六条区间跨度均 ≤ 6：枚举行列式为正的基向量组合（≤ 7⁴，先按 det>0 过滤），
对每个基向量对用“标记必须存在原点使其落入容差窗”做可行性预筛，再枚举原点
（≤ 7²）。每组参数下，用 MRV 深度优先 + 二分图最大匹配前向检查 + 字典序
下界剪枝，一次搜索 0..2 个弃点的全部互异分配。全程整数运算，结果确定可复现。

## 运行

```bash
# 可配置宿主机端口
API_PORT=8080 docker compose up --build api
curl localhost:8080/health
```

## 一次性校验

```bash
./scripts/verify                 # 有 docker：构建+起服务+健康后校验+自动清理
API_PORT=9000 ./scripts/verify   # 无 docker：自动使用 .venv 本地校验
```

校验流程：等待 `/health` 通过 → `pytest`（求解器 + API 共 20 个测试，含独立
暴力枚举对照最优性）→ 针对运行服务的 5 组冒烟（漏读+双杂点、仅漏读、
三杂点无解、区间跨度矛盾 422、重复编号 422）→ 以退出码汇报（0 通过）。

也可用 compose 的一次性服务（依赖 api 健康后才启动，退出码即校验结果）：

```bash
docker compose up --abort-on-container-exit --exit-code-from verify verify
```

## 项目结构

```
app/solver.py    精确求解器（枚举 + 分支定界互异分配）
app/models.py    请求/响应模型与输入校验
app/main.py      FastAPI 路由（/health、/api/wafer-grids/reconstruct）
tests/           pytest 测试
scripts/smoke.py 端到端冒烟（仅用标准库）
scripts/verify   一次性校验入口
Dockerfile, docker-compose.yml, requirements.txt
```
