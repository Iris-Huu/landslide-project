"""
比较 Satellite001 到 UAV0001、UAV0011 的两次相似变换。

将第一次变换按无人机网格关系换算到 UAV0011 坐标系，
再与第二次变换进行比较。

注意：
- 这是模型间的一致性检查，不是独立配准精度；
- 两次匹配来自相同场景和同一卫星切片；
- 本程序不读取或修改原始影像。
"""

from pathlib import Path
import csv
import json

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[2]

REG_DIR = PROJECT_DIR / "data/interim/registration"
GRID_PATH = (
    PROJECT_DIR
    / "data/interim/reconstructed_grids"
    / "uav_grid_coordinates_v2.csv"
)

FIRST_PATH = REG_DIR / "moxitaidi_sift_sample_transform.json"
SECOND_PATH = REG_DIR / "sat001_uav0011_result.json"
OUTPUT_PATH = REG_DIR / "sat001_transform_consistency.json"

FIRST_UAV = "luding_UAV0001.tif"
SECOND_UAV = "luding_UAV0011.tif"
SATELLITE = "Moxitaidi_Satellite001.tif"

TILE_SIZE = 512
GRID_STEP = 256
SAMPLE_COUNT = 21


def read_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def apply_transform(points, matrix):
    """将一组 (x, y) 点应用 2×3 变换矩阵。"""
    return points @ matrix[:, :2].T + matrix[:, 2]


def main():
    if OUTPUT_PATH.exists():
        raise SystemExit(
            f"输出文件已存在，为避免覆盖而停止：{OUTPUT_PATH}"
        )

    first = read_json(FIRST_PATH)
    second = read_json(SECOND_PATH)

    # 核对两份结果确实来自预期的影像
    if (
        first["source_image"] != SATELLITE
        or first["target_image"] != FIRST_UAV
    ):
        raise ValueError("第一份变换记录的影像名称不符合预期。")

    if (
        second["satellite_image"] != SATELLITE
        or second["uav_image"] != SECOND_UAV
    ):
        raise ValueError("第二份变换记录的影像名称不符合预期。")

    if second["transform_satellite_to_uav"] is None:
        raise ValueError("第二份记录没有可用的变换矩阵。")

    first_matrix = np.asarray(first["matrix_2x3"], dtype=float)
    second_matrix = np.asarray(
        second["transform_satellite_to_uav"],
        dtype=float,
    )

    if first_matrix.shape != (2, 3) or second_matrix.shape != (2, 3):
        raise ValueError("变换矩阵不是预期的 2×3 形状。")

    with GRID_PATH.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    first_rows = [
        row for row in rows if row["filename"] == FIRST_UAV
    ]
    second_rows = [
        row for row in rows if row["filename"] == SECOND_UAV
    ]

    if len(first_rows) != 1 or len(second_rows) != 1:
        raise ValueError("无人机坐标表中的样本记录不唯一。")

    first_grid = first_rows[0]
    second_grid = second_rows[0]

    if first_grid["component"] != second_grid["component"]:
        raise ValueError(
            "两张无人机切片不在同一连通组，不能直接比较坐标。"
        )

    delta_row = (
        int(second_grid["grid_row"])
        - int(first_grid["grid_row"])
    )
    delta_col = (
        int(second_grid["grid_col"])
        - int(first_grid["grid_col"])
    )

    # x 对应列，y 对应行
    grid_offset_xy = np.array([
        delta_col * GRID_STEP,
        delta_row * GRID_STEP,
    ], dtype=float)

    # 在 512×512 的卫星切片上均匀选点
    coordinates = np.linspace(
        0, TILE_SIZE - 1, SAMPLE_COUNT
    )
    x_values, y_values = np.meshgrid(
        coordinates, coordinates
    )
    satellite_points = np.column_stack([
        x_values.ravel(),
        y_values.ravel(),
    ])

    # 方法一：卫星 → UAV0001，再换算到 UAV0011
    predicted_via_first = (
        apply_transform(satellite_points, first_matrix)
        - grid_offset_xy
    )

    # 方法二：卫星 → UAV0011
    predicted_direct = apply_transform(
        satellite_points,
        second_matrix,
    )

    # 只比较第二次变换落在 UAV0011 画布内的点。
    # 这只是几何范围筛选，不代表这些点都是有效影像。
    inside = (
        (predicted_direct[:, 0] >= 0)
        & (predicted_direct[:, 0] < TILE_SIZE)
        & (predicted_direct[:, 1] >= 0)
        & (predicted_direct[:, 1] < TILE_SIZE)
    )

    if not inside.any():
        raise SystemExit(
            "采样点没有落入 UAV0011 几何画布，无法进行比较。"
        )

    differences = np.linalg.norm(
        predicted_via_first[inside] - predicted_direct[inside],
        axis=1,
    )

    result = {
        "satellite_image": SATELLITE,
        "first_uav_image": FIRST_UAV,
        "second_uav_image": SECOND_UAV,
        "uav_grid_delta_row": delta_row,
        "uav_grid_delta_col": delta_col,
        "uav_grid_offset_xy_pixels": grid_offset_xy.tolist(),
        "satellite_sampling_grid": f"{SAMPLE_COUNT}x{SAMPLE_COUNT}",
        "sample_points_inside_second_uav_canvas": int(inside.sum()),
        "difference_median_uav_pixels": float(np.median(differences)),
        "difference_max_uav_pixels": float(np.max(differences)),
        "note": (
            "Comparison of two estimated models, not independent "
            "registration accuracy. First transform is extrapolated."
        ),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("x", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    print(
        f"无人机网格位移：行 {delta_row:+d}，"
        f"列 {delta_col:+d}"
    )
    print(
        "UAV0011 相对 UAV0001 的像素偏移 "
        f"(x, y)：{grid_offset_xy.tolist()}"
    )
    print(
        f"参与比较的卫星采样点：{inside.sum()} 个 "
        f"（总采样 {len(satellite_points)} 个）"
    )
    print(
        f"两种预测位置差的中位数："
        f"{np.median(differences):.2f} 个无人机像素"
    )
    print(
        f"两种预测位置差的最大值："
        f"{np.max(differences):.2f} 个无人机像素"
    )
    print(f"\n结果已保存：{OUTPUT_PATH}")
    print(
        "注意：这是两个模型的相互比较，"
        "不等于独立测得的真实配准误差。"
    )


if __name__ == "__main__":
    main()
