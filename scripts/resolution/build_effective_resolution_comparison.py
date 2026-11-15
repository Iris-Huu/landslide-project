from pathlib import Path
import csv
import json
import numpy as np

ROOT = Path("/root/autodl-tmp/landslide_project")
AUDIT = ROOT / "reports/tables/effective_resolution_audit_v1.csv"
MODEL_0519 = ROOT / "data/interim/registration/sat100_uav0519_anchor_check_result.json"
MODEL_0533 = ROOT / "data/interim/registration/sat100_uav0533_neighbor_result.json"

OUT_CSV = ROOT / "reports/tables/effective_resolution_comparison_v1.csv"
OUT_MD = ROOT / "reports/effective_resolution_comparison_v1.md"

with AUDIT.open(encoding="utf-8-sig", newline="") as f:
    audit_rows = list(csv.DictReader(f))

sigma_values = [
    float(r["best_sigma_uav_pixels"])
    for r in audit_rows
    if r["best_sigma_uav_pixels"] not in ("", "not_determined")
]

scale_values = []

for path in (MODEL_0519, MODEL_0533):
    model = json.loads(path.read_text(encoding="utf-8"))
    matrix = np.asarray(model["transform_satellite_to_uav"], dtype=float)
    scale_values.append(float(np.linalg.norm(matrix[:, 0])))

scale_min = min(scale_values)
scale_max = max(scale_values)

sigma_min = min(sigma_values)
sigma_max = max(sigma_values)

rows = [
    {
        "dataset": "CAS_Moxitaidi_SAT",
        "nominal_gsd_m_per_px": "0.6",
        "nominal_gsd_evidence": "CAS_paper_Table2_reported_0.6m_conflicts_with_Sentinel2_native_resolution",
        "relative_pixel_scale": f"{scale_min:.6f}-{scale_max:.6f}",
        "relative_scale_basis": "SAT100_to_UAV0519_and_UAV0533_local_affine_models",
        "exploratory_blur_sigma_uav_px": "",
        "absolute_effective_resolution_m": "",
        "effective_resolution_status": "not_determined",
        "limitation": "Paper_reported_0.6m_not_verified_as_true_GSD_and_sensor_native_resolution_conflict",
    },
    {
        "dataset": "CAS_Moxitaidi_UAV_0.6m",
        "nominal_gsd_m_per_px": "0.6",
        "nominal_gsd_evidence": "CAS_archive_name_UAV-0.6m_not_independently_verified",
        "relative_pixel_scale": "1.0_reference_UAV_pixel",
        "relative_scale_basis": "UAV_grid_coordinate_system",
        "exploratory_blur_sigma_uav_px": f"{sigma_min:.2f}-{sigma_max:.2f}",
        "absolute_effective_resolution_m": "",
        "effective_resolution_status": "not_determined",
        "limitation": "Blur_sigma_is_not_effective_GSD_and_matching_is_exploratory",
    },
]

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

lines = [
    "# Moxitaidi有效分辨率证据对照表",
    "",
    "## 1. 当前证据",
    "",
    "| 数据集 | 标称GSD (m/px) | 相对几何尺度 | 探索性模糊参数 | 绝对有效分辨率 |",
    "|---|---|---|---|---|",
    (
        f"| Moxitaidi SAT | 0.6（论文报告，存在来源疑问） | {scale_min:.4f}–{scale_max:.4f} "
        "UAV px/SAT px | 未测 | 未确定 |"
    ),
    (
        f"| Moxitaidi UAV-0.6m | 0.6（仅数据包名称） | 参考坐标系 | "
        f"σ={sigma_min:.1f}–{sigma_max:.1f} UAV px | 未确定 |"
    ),
    "",
    "## 2. 证据说明",
    "",
    "- SAT001、SAT100、UAV0519、UAV0533的TIFF均无有效CRS、GCP和RPC。",
    "- Rasterio报告的单位变换(1,1)不能解释为米/像素。",
    "- SAT TIFF中的96 DPI是显示/打印标签，不代表地面采样距离。",
    "- 17个CAS ZIP包未检索到文件名明显属于元数据或说明文档的条目。",
    "- UAV-0.6m来自压缩包名称，尚未核查独立发布方证据。",
    "- SAT100配准尺度来自两个局部SIFT/RANSAC仿射模型。",
    "- 模糊参数来自SAT001相关的4个探索性窗口，不能视为真实有效GSD。",
    "- 这些模糊窗口尚不能作为已验证的同址配准测量依据。",
    "",
    "## 3. 结论与待补充证据",
    "",
    "当前可以报告数据集命名所暗示的标称GSD及局部相对像素尺度，",
    "但不能可靠地确定SAT和UAV的绝对有效地面分辨率。",
    "需要进一步核查数据集论文、官方元数据或具有绝对尺度的标定证据。",
    "",
]

OUT_MD.write_text("\n".join(lines), encoding="utf-8")

print("===== 第100步：有效分辨率对照表 =====")
for row in rows:
    print(
        row["dataset"],
        "| 标称GSD:", row["nominal_gsd_m_per_px"] or "未核实",
        "| 有效分辨率:", row["effective_resolution_status"],
    )

print(f"局部相对尺度：{scale_min:.6f} - {scale_max:.6f}")
print(f"探索性sigma：{sigma_min:.2f} - {sigma_max:.2f} UAV px")
print("CSV保存：", OUT_CSV)
print("报告保存：", OUT_MD)
