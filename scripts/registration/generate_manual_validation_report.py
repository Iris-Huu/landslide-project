from pathlib import Path
import csv
import numpy as np

root = Path("/root/autodl-tmp/landslide_project")
table_dir = root / "reports/tables"

files = {
    "UAV0519": table_dir / "sat100_uav0519_manual_validation_v1.csv",
    "UAV0533": table_dir / "sat100_uav0533_manual_validation_v1.csv",
}

report_path = root / "reports/sat100_cross_platform_registration_validation_v1.md"

sections = [
    "# SAT100跨平台局部配准人工验证报告",
    "",
    "## 1. 实验目的",
    "",
    "验证已有SAT100至UAV0519和UAV0533的配准矩阵与人工同名点的几何一致性。",
    "",
    "## 2. 人工核验结果",
    "",
    "| 配准组合 | 点数 | RMSE (px) | 中位误差 (px) | 最大误差 (px) |",
    "|---|---:|---:|---:|---:|",
]

for name, path in files.items():
    if not path.exists():
        raise FileNotFoundError(f"缺少文件：{path}")

    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if name == "UAV0519":
        rows = [r for r in rows if int(r["round"]) == 2]

    if not rows:
        raise ValueError(f"{name}没有可用控制点")

    errors = np.array([float(r["error_px"]) for r in rows])

    rmse = float(np.sqrt(np.mean(errors ** 2)))
    median = float(np.median(errors))
    maximum = float(np.max(errors))

    sections.append(
        f"| SAT100 → {name} | {len(rows)} | "
        f"{rmse:.2f} | {median:.2f} | {maximum:.2f} |"
    )

sections.extend([
    "",
    "## 3. 验证方法",
    "",
    "- 使用此前保存的SIFT/RANSAC局部配准矩阵。",
    "- 使用人工鼠标标注的SAT与UAV同名地物坐标。",
    "- 本次人工点仅用于核验，不重新拟合配准矩阵。",
    "- 误差采用UAV像素坐标系中的二维欧氏距离。",
    "- UAV0519有两轮标注，汇总使用第二轮，第一轮数据应完整保留。",
    "",
    "## 4. 适用范围与局限",
    "",
    "- 结果支持SAT100与两张相邻UAV影像具有局部同址配准可行性。",
    "- 人工点可能存在标注偏差。",
    "- 尚未排除人工点与原始自动匹配内点重合的可能。",
    "- 这是探索性人工核验，不是严格独立的测试精度。",
    "- 当前结果不能证明整个CAS数据集已经完成跨平台地理配准。",
    "- 当前结果不能证明训练、验证、测试之间不存在跨平台空间泄漏。",
    "",
    "## 5. 阶段性结论",
    "",
    "SAT100到UAV0519、UAV0533的局部配准变换均获得人工点支持。",
    "但该结论仅适用于当前核验的局部影像组合。",
    "",
])

report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text("\n".join(sections), encoding="utf-8")

print("\n".join(sections))
print("\n报告保存成功：", report_path)
