from pathlib import Path
import csv
import json
import numpy as np

ROOT = Path("/root/autodl-tmp/landslide_project")
REG = ROOT / "data/interim/registration"
TABLE = ROOT / "reports/tables"

model_files = {
    "0518": REG / "sat100_uav0518_neighbor_result.json",
    "0519": REG / "sat100_uav0519_anchor_check_result.json",
    "0533": REG / "sat100_uav0533_neighbor_result.json",
}

manual_files = {
    "0519": TABLE / "sat100_uav0519_manual_validation_v1.csv",
    "0533": TABLE / "sat100_uav0533_manual_validation_v1.csv",
}

grid = {
    "0518": (42, 33),
    "0519": (42, 34),
    "0533": (43, 34),
}

points = np.array([
    [0, 0, 1],
    [511, 0, 1],
    [0, 511, 1],
    [511, 511, 1],
    [255.5, 255.5, 1],
], dtype=float)

models = {}
global_predictions = {}

for name, path in model_files.items():
    data = json.loads(path.read_text(encoding="utf-8"))
    M = np.asarray(data["transform_satellite_to_uav"], dtype=float)

    models[name] = data
    row, col = grid[name]
    global_predictions[name] = (
        (M @ points.T).T + np.array([col * 256, row * 256])
    )

manual_rmse = {}
manual_n = {}

for name, path in manual_files.items():
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    if name == "0519":
        rows = [r for r in rows if int(r["round"]) == 2]

    errors = np.array([float(r["error_px"]) for r in rows])
    manual_rmse[name] = float(np.sqrt(np.mean(errors**2)))
    manual_n[name] = len(errors)

def consistency(a, b):
    distances = np.linalg.norm(
        global_predictions[a] - global_predictions[b], axis=1
    )
    return float(distances[-1]), float(distances.max())

rows = []

for name in ["0519", "0533", "0518"]:
    M = np.asarray(
        models[name]["transform_satellite_to_uav"], dtype=float
    )
    scale = float(np.linalg.norm(M[:, 0]))

    reference = "0533" if name == "0519" else "0519"
    center_diff, max_diff = consistency(name, reference)

    if name == "0519":
        grade = "A_candidate"
        decision = "local_feasibility_supported"
    elif name == "0533":
        grade = "B_candidate"
        decision = "local_feasibility_supported_with_caution"
    else:
        grade = "review_required"
        decision = "not_approved"

    rows.append({
        "sat_image": "Moxitaidi_Satellite100.tif",
        "uav_image": f"luding_UAV{name}.tif",
        "ransac_inliers": models[name].get("ransac_inliers"),
        "scale": round(scale, 6),
        "manual_points": manual_n.get(name, 0),
        "manual_rmse_px": (
            round(manual_rmse[name], 4)
            if name in manual_rmse else ""
        ),
        "reference_uav": f"UAV{reference}",
        "center_difference_px": round(center_diff, 4),
        "max_difference_px": round(max_diff, 4),
        "evidence_grade": grade,
        "decision": decision,
        "training_pair_approved": False,
        "independent_validation_confirmed": False,
    })

out_csv = REG / "sat100_registration_reliability_decision_v1.csv"

with out_csv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

out_md = ROOT / "reports/sat100_registration_reliability_decision_v1.md"

lines = [
    "# SAT100局部跨平台配准可靠性判定",
    "",
    "## 1. 证据分级",
    "",
    "| 配准组合 | 等级 | RANSAC内点 | 人工RMSE(px) | 中心一致性差(px) |",
    "|---|---|---:|---:|---:|",
]

for r in rows:
    lines.append(
        f"| SAT100 → {r['uav_image']} | {r['evidence_grade']} | "
        f"{r['ransac_inliers']} | {r['manual_rmse_px'] or '未验证'} | "
        f"{r['center_difference_px']} |"
    )

lines.extend([
    "",
    "## 2. 判定依据",
    "",
    "- A/B仅表示局部探索证据的相对强弱，并非正式精度认证。",
    "- UAV0519和UAV0533模型相互一致，但仍存在人工核验误差。",
    "- UAV0518与其他模型尺度及几何位置不一致，暂不批准。",
    "- 人工点尚未证明独立于原SIFT/RANSAC拟合内点。",
    "- 当前尚未证明SAT与UAV全区域无空间泄漏。",
    "- 所有组合的training_pair_approved均为False。",
    "",
    "## 3. 阶段结论",
    "",
    "SAT100与UAV0519、UAV0533具有局部同址配准可行性证据。",
    "目前仅作为方法可行性研究，不作为正式无泄漏训练配对。",
    "UAV0518保持待复核状态。",
    "",
])

out_md.write_text("\n".join(lines), encoding="utf-8")

print("===== 配准可靠性判定 =====")
for r in rows:
    print(
        r["uav_image"],
        "等级:", r["evidence_grade"],
        "人工RMSE:", r["manual_rmse_px"],
        "中心差:", r["center_difference_px"],
        "训练批准:", r["training_pair_approved"],
    )

print("\nCSV保存：", out_csv)
print("报告保存：", out_md)
