"""
估算 Satellite001 与 UAV0001 所在网格组的几何重叠。

使用：
- 已保存的 Satellite001 → UAV0001 相似变换；
- UAV v2 网格中的相对坐标。

重要限制：
- 计算的是矩形几何范围，不是有效影像范围；
- 局部变换向外推算存在不确定性；
- 只检查 UAV0001 所在连通组；
- 本程序用于排查风险，不生成数据划分。
"""

from pathlib import Path
import csv
import json

import cv2
import numpy as np


PROJECT = Path(__file__).resolve().parents[2]

TRANSFORM_PATH = (
    PROJECT
    / "data/interim/registration"
    / "moxitaidi_sift_sample_transform.json"
)

GRID_PATH = (
    PROJECT
    / "data/interim/reconstructed_grids"
    / "uav_grid_coordinates_v2.csv"
)

OUTPUT_PATH = (
    PROJECT
    / "reports/tables"
    / "sat001_uav_group_geometric_overlap.csv"
)

SATELLITE_NAME = "Moxitaidi_Satellite001.tif"
REFERENCE_UAV = "luding_UAV0001.tif"

TILE_SIZE = 512
GRID_STEP = 256


def main():
    if OUTPUT_PATH.exists():
        raise SystemExit(
            f"结果文件已存在，未覆盖：{OUTPUT_PATH}"
        )

    with TRANSFORM_PATH.open(encoding="utf-8") as file:
        transform = json.load(file)

    if (
        transform["source_image"] != SATELLITE_NAME
        or transform["target_image"] != REFERENCE_UAV
    ):
        raise ValueError("变换记录中的影像名称不符合预期。")

    matrix = np.asarray(transform["matrix_2x3"], dtype=np.float64)

    if matrix.shape != (2, 3):
        raise ValueError(f"变换矩阵形状异常：{matrix.shape}")

    with GRID_PATH.open(
        encoding="utf-8-sig", newline=""
    ) as file:
        rows = list(csv.DictReader(file))

    reference_rows = [
        row for row in rows
        if row["filename"] == REFERENCE_UAV
    ]

    if len(reference_rows) != 1:
        raise ValueError("参考无人机影像在坐标表中的记录不唯一。")

    reference = reference_rows[0]
    group = reference["component"]
    ref_row = int(reference["grid_row"])
    ref_col = int(reference["grid_col"])

    group_rows = [
        row for row in rows
        if row["component"] == group
    ]

    # 卫星切片的几何边界，使用像素边缘坐标
    satellite_corners = np.array([
        [0, 0],
        [TILE_SIZE, 0],
        [TILE_SIZE, TILE_SIZE],
        [0, TILE_SIZE],
    ], dtype=np.float64)

    # 转换到 UAV0001 的像素坐标系
    satellite_polygon = (
        satellite_corners @ matrix[:, :2].T
        + matrix[:, 2]
    ).astype(np.float32)

    results = []

    for row in group_rows:
        delta_row = int(row["grid_row"]) - ref_row
        delta_col = int(row["grid_col"]) - ref_col

        # 无人机切片左上角在 UAV0001 坐标系中的位置
        x = delta_col * GRID_STEP
        y = delta_row * GRID_STEP

        uav_polygon = np.array([
            [x, y],
            [x + TILE_SIZE, y],
            [x + TILE_SIZE, y + TILE_SIZE],
            [x, y + TILE_SIZE],
        ], dtype=np.float32)

        area, _ = cv2.intersectConvexConvex(
            satellite_polygon,
            uav_polygon,
        )

        area = max(float(area), 0.0)

        results.append({
            "satellite": SATELLITE_NAME,
            "uav_image": row["filename"],
            "uav_component": group,
            "delta_row": delta_row,
            "delta_col": delta_col,
            "geometric_overlap_pixels": area,
            "uav_tile_overlap_percent": (
                area / (TILE_SIZE * TILE_SIZE) * 100
            ),
        })

    results.sort(
        key=lambda item: (
            -item["uav_tile_overlap_percent"],
            item["uav_image"],
        )
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open(
        "x", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "satellite",
                "uav_image",
                "uav_component",
                "delta_row",
                "delta_col",
                "geometric_overlap_pixels",
                "uav_tile_overlap_percent",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    positive = [
        row for row in results
        if row["geometric_overlap_pixels"] > 0
    ]

    print(f"无人机连通组：{group}")
    print(f"检查的无人机切片：{len(results)} 张")
    print(f"估计几何交集大于 0：{len(positive)} 张")
    print("\n交集比例最高的前 10 张：")

    for row in positive[:10]:
        print(
            f"  {row['uav_image']}："
            f"{row['uav_tile_overlap_percent']:.2f}%"
        )

    print(f"\n结果已保存：{OUTPUT_PATH}")
    print(
        "注意：结果是局部变换下的矩形几何估计，"
        "不是有效影像覆盖或安全划分结论。"
    )


if __name__ == "__main__":
    main()