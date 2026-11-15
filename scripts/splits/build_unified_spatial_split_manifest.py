from pathlib import Path
from collections import Counter
import csv

ROOT = Path("/root/autodl-tmp/landslide_project")

SAT_GRID = ROOT / "data/interim/reconstructed_grids/satellite_grid_coordinates_v2.csv"
UAV_GRID = ROOT / "data/interim/reconstructed_grids/uav_grid_coordinates_v2.csv"
UAV_SPLIT = ROOT / "data/interim/splits/moxitaidi_uav_split_candidate_v1.csv"
SAT_RISK = ROOT / "data/interim/splits/satellite_leakage_risk_v1.csv"

OUTPUT = ROOT / "data/interim/splits/moxitaidi_unified_spatial_manifest_v1.csv"
REPORT = ROOT / "reports/moxitaidi_unified_spatial_manifest_validation_v1.md"

def read_rows(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

sat_grid = read_rows(SAT_GRID)
uav_grid = read_rows(UAV_GRID)
uav_split = read_rows(UAV_SPLIT)
sat_risk = read_rows(SAT_RISK)

def unique_index(rows, key):
    index = {}
    for row in rows:
        name = row[key]
        if name in index:
            raise ValueError(f"重复记录：{name}")
        index[name] = row
    return index

sat_risk_by_name = unique_index(sat_risk, "satellite_image")
uav_split_by_name = unique_index(uav_split, "filename")

sat_names = {r["filename"] for r in sat_grid}
uav_names = {r["filename"] for r in uav_grid}

assert len(sat_grid) == 652, "SAT图块数量异常"
assert len(uav_grid) == 984, "UAV图块数量异常"
assert sat_names == set(sat_risk_by_name), "SAT风险表与网格不一致"
assert uav_names == set(uav_split_by_name), "UAV划分表与网格不一致"
assert not sat_names & uav_names, "SAT与UAV文件名发生冲突"

allowed_splits = {
    "train_candidate",
    "val_candidate",
    "test_candidate",
    "buffer",
    "unassigned",
}

rows = []

for r in sat_grid:
    risk = sat_risk_by_name[r["filename"]]

    if str(risk["approved_for_training_by_this_check"]) != "0":
        raise ValueError(f"SAT批准状态异常：{r['filename']}")

    rows.append({
        "platform": "SAT",
        "filename": r["filename"],
        "component": r["component"],
        "grid_row": r["grid_row"],
        "grid_col": r["grid_col"],
        "split": "unassigned",
        "within_platform_candidate": False,
        "cross_platform_leakage_verified": False,
        "approved_for_training": False,
        "approved_for_evaluation": False,
        "risk_reason": "SAT_UAV_spatial_overlap_not_excluded",
    })

for r in uav_grid:
    candidate = uav_split_by_name[r["filename"]]
    split = candidate["split"]

    if split not in allowed_splits:
        raise ValueError(f"未知划分：{split}")

    for key in ("component", "grid_row", "grid_col"):
        if r[key] != candidate[key]:
            raise ValueError(
                f"UAV网格与划分不一致：{r['filename']}，字段：{key}"
            )

    rows.append({
        "platform": "UAV",
        "filename": r["filename"],
        "component": r["component"],
        "grid_row": r["grid_row"],
        "grid_col": r["grid_col"],
        "split": split,
        "within_platform_candidate": split in {
            "train_candidate", "val_candidate", "test_candidate"
        },
        "cross_platform_leakage_verified": False,
        "approved_for_training": False,
        "approved_for_evaluation": False,
        "risk_reason": (
            "buffer_excluded" if split == "buffer"
            else "outside_active_split" if split == "unassigned"
            else "within_UAV_candidate_only_cross_platform_not_verified"
        ),
    })

assert len(rows) == 1636, "最终记录数量不正确"
assert len({(r["platform"], r["filename"]) for r in rows}) == 1636

counts = Counter((r["platform"], r["split"]) for r in rows)

assert counts[("SAT", "unassigned")] == 652
assert counts[("UAV", "train_candidate")] == 484
assert counts[("UAV", "val_candidate")] == 71
assert counts[("UAV", "test_candidate")] == 149
assert counts[("UAV", "buffer")] == 57
assert counts[("UAV", "unassigned")] == 223

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

active = 484 + 71 + 149

report = [
    "# Moxitaidi统一空间划分候选清单校验",
    "",
    "## 1. 完整性",
    "",
    f"- SAT图块：{len(sat_grid)}",
    f"- UAV图块：{len(uav_grid)}",
    f"- 总记录数：{len(rows)}",
    "- 每张图块在统一清单中恰好出现一次。",
    "",
    "## 2. UAV划分",
    "",
    "| 状态 | 数量 |",
    "|---|---:|",
]

for status in [
    "train_candidate", "val_candidate", "test_candidate",
    "buffer", "unassigned"
]:
    report.append(f"| {status} | {counts[('UAV', status)]} |")

report += [
    "",
    "## 3. 有效候选划分比例",
    "",
    f"- Train：{484 / active * 100:.2f}%",
    f"- Val：{71 / active * 100:.2f}%",
    f"- Test：{149 / active * 100:.2f}%",
    "",
    "## 4. 风险与使用限制",
    "",
    "- 仅UAV最大连通分量的704张图块进入训练/验证/测试候选集。",
    "- UAV缓冲区57张、未分配223张均不得自动进入候选训练或测试。",
    "- SAT全部652张暂不分配训练/验证/测试角色。",
    "- 当前未证实所有SAT与UAV图块之间的空间无泄漏。",
    "- 此文件是候选划分清单，不是已经冻结批准的全平台实验划分。",
    "- 本清单全部approved_for_training和approved_for_evaluation均为False。",
    "",
]

REPORT.write_text("\n".join(report), encoding="utf-8")

print("===== 统一划分清单生成成功 =====")
print("SAT图块：", len(sat_grid))
print("UAV图块：", len(uav_grid))
print("总记录：", len(rows))
print("UAV划分：", dict(
    (k[1], v) for k, v in counts.items() if k[0] == "UAV"
))
print("正式批准训练数量：", sum(r["approved_for_training"] for r in rows))
print("CSV：", OUTPUT)
print("报告：", REPORT)
