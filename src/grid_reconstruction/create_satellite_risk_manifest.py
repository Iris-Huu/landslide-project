"""
根据现有局部检查，生成 Moxitaidi SAT 切片的风险状态清单。

本清单只记录证据状态，不判定任何卫星切片可以安全用于训练。
"""

import csv
import math
from collections import Counter
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]

SAT_GRID = (
    PROJECT / "data/interim/reconstructed_grids/"
    "satellite_grid_coordinates_v2.csv"
)
LOCAL_CHECK = (
    PROJECT / "data/interim/registration/"
    "sat100_uav_group2_overlap_v1.csv"
)
OUTPUT = (
    PROJECT / "data/interim/splits/"
    "satellite_leakage_risk_v1.csv"
)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    if OUTPUT.exists():
        raise SystemExit(f"结果文件已存在，未覆盖：{OUTPUT}")

    satellite_rows = read_csv(SAT_GRID)
    local_rows = read_csv(LOCAL_CHECK)

    names = [row["filename"] for row in satellite_rows]

    if len(names) != 652 or len(set(names)) != 652:
        raise SystemExit("卫星坐标表数量异常或存在重复文件名。")

    if "Moxitaidi_Satellite100.tif" not in names:
        raise SystemExit("卫星坐标表中没有找到 Satellite100。")

    if len(local_rows) != 761:
        raise SystemExit(
            f"SAT100 局部检查记录数量异常：{len(local_rows)}"
        )

    if len({row["uav_image"] for row in local_rows}) != len(local_rows):
        raise SystemExit("SAT100 局部检查存在重复 UAV 文件名。")

    overlap_counts = Counter()
    expanded_counts = Counter()

    for row in local_rows:
        if row["satellite"] != "Moxitaidi_Satellite100.tif":
            raise SystemExit("局部结果中出现非 SAT100 记录。")

        area = float(row["geometric_overlap_pixels"])

        if not math.isfinite(area) or area < 0:
            raise SystemExit("局部几何交集数值异常。")

        if row["inside_expanded_bbox"] not in {"0", "1"}:
            raise SystemExit("外扩框字段异常。")

        if area > 0:
            overlap_counts[row["split"]] += 1

        if row["inside_expanded_bbox"] == "1":
            expanded_counts[row["split"]] += 1

    manifest = []

    for row in satellite_rows:
        name = row["filename"]

        if name == "Moxitaidi_Satellite100.tif":
            status = "local_anchor_only_requires_review"
            note = (
                "Local model checked against UAV component 2; "
                "other components and global registration remain unverified."
            )
        else:
            status = "unverified"
            note = (
                "No complete satellite-to-UAV test-area "
                "separation verification available."
            )

        manifest.append({
            "satellite_image": name,
            "satellite_component": row["component"],
            "screening_status": status,
            "approved_for_training_by_this_check": 0,
            "note": note,
        })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)

    statuses = Counter(row["screening_status"] for row in manifest)

    print(f"卫星影像：{len(manifest)} 张")
    print("状态统计：")
    for status, count in sorted(statuses.items()):
        print(f"  {status}: {count}")

    print("\nSAT100 与 UAV 第 2 组的已保存局部估计：")
    for split in [
        "train_candidate",
        "val_candidate",
        "test_candidate",
        "buffer",
    ]:
        print(
            f"  {split}: 几何交集 {overlap_counts[split]} 张；"
            f"外扩框相交 {expanded_counts[split]} 张"
        )

    print(f"\n清单已保存：{OUTPUT}")
    print("注意：该清单没有批准任何影像进入正式训练集。")


if __name__ == "__main__":
    main()
