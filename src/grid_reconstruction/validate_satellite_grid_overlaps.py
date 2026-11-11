"""
验证 UAV v2 网格内按位置推测会重叠的切片。

比较水平、垂直和两个对角方向的重叠像素。
本程序不检查不同连通组之间是否存在重叠。
"""

from pathlib import Path
import csv
import warnings

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning


PROJECT = Path(__file__).resolve().parents[2]

IMAGE_DIR = (
    PROJECT / "data" / "raw" / "CAS"
    / "Moxitaidi_SAT" / "img"
)

GRID_DIR = (
    PROJECT / "data" / "interim" / "reconstructed_grids"
)

COORD_PATH = GRID_DIR / "satellite_grid_coordinates_v2.csv"
OUTPUT_PATH = GRID_DIR / "satellite_grid_overlap_validation_v2.csv"

warnings.filterwarnings(
    "ignore",
    category=NotGeoreferencedWarning,
)


def read_image(filename):
    """读取一张三通道切片。"""
    with rasterio.open(IMAGE_DIR / filename) as src:
        image = src.read()

    if image.shape != (3, 512, 512):
        raise ValueError(f"{filename} 形状异常：{image.shape}")

    return image


def overlap_regions(source, target, dr, dc):
    """
    根据目标相对源的网格位移，取出对应重叠区域。

    dr：行位移；dc：列位移。
    每个网格单位对应 256 像素。
    """
    row_shift = dr * 256
    col_shift = dc * 256

    source_row_start = max(row_shift, 0)
    source_row_end = min(512 + row_shift, 512)
    source_col_start = max(col_shift, 0)
    source_col_end = min(512 + col_shift, 512)

    target_row_start = max(-row_shift, 0)
    target_row_end = min(512 - row_shift, 512)
    target_col_start = max(-col_shift, 0)
    target_col_end = min(512 - col_shift, 512)

    source_part = source[
        :,
        source_row_start:source_row_end,
        source_col_start:source_col_end,
    ]
    target_part = target[
        :,
        target_row_start:target_row_end,
        target_col_start:target_col_end,
    ]

    return source_part, target_part


def main():
    if OUTPUT_PATH.exists():
        raise SystemExit(
            f"输出文件已存在，为避免覆盖而停止：{OUTPUT_PATH}"
        )

    with COORD_PATH.open(
        encoding="utf-8", newline=""
    ) as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise SystemExit("坐标表为空。")

    # (连通组, 行, 列) -> 影像名称
    positions = {}
    seen_filenames = set()

    for row in rows:
        filename = row["filename"]
        key = (
            int(row["component"]),
            int(row["grid_row"]),
            int(row["grid_col"]),
        )

        if filename in seen_filenames:
            raise ValueError(f"坐标表中有重复影像：{filename}")

        if key in positions:
            raise ValueError(
                f"同一网格位置存在多张影像："
                f"{positions[key]} / {filename}"
            )

        seen_filenames.add(filename)
        positions[key] = filename

    # 只检查这四个方向，避免反向重复计算同一对
    directions = [
        ("right", 0, 1),
        ("down", 1, 0),
        ("down_right", 1, 1),
        ("down_left", 1, -1),
    ]

    pairs = []

    for (component, row, col), source_name in positions.items():
        for direction, dr, dc in directions:
            target_key = (component, row + dr, col + dc)
            target_name = positions.get(target_key)

            if target_name is not None:
                pairs.append({
                    "component": component,
                    "source_image": source_name,
                    "target_image": target_name,
                    "direction": direction,
                    "dr": dr,
                    "dc": dc,
                })

    print(f"坐标表影像：{len(rows)} 张")
    print(f"需要检查的组内重叠对：{len(pairs)} 对", flush=True)

    results = []
    errors = []

    for index, pair in enumerate(pairs, start=1):
        source_name = pair["source_image"]
        target_name = pair["target_image"]

        try:
            source = read_image(source_name)
            target = read_image(target_name)

            source_part, target_part = overlap_regions(
                source,
                target,
                pair["dr"],
                pair["dc"],
            )

            if (
                source_part.shape != target_part.shape
                or source_part.size == 0
            ):
                raise ValueError("重叠区域尺寸异常")

            # 任一通道不同，即该位置的像素不同
            different_pixels = np.any(
                source_part != target_part,
                axis=0,
            )

            results.append({
                "component": pair["component"],
                "source_image": source_name,
                "target_image": target_name,
                "direction": pair["direction"],
                "overlap_pixels": int(different_pixels.size),
                "different_pixels": int(different_pixels.sum()),
                "status": (
                    "identical"
                    if not different_pixels.any()
                    else "different"
                ),
            })

        except Exception as error:
            errors.append((source_name, target_name, str(error)))

        if index % 500 == 0 or index == len(pairs):
            print(f"  已检查 {index}/{len(pairs)} 对", flush=True)

    if errors:
        print(f"\n检查失败：{len(errors)} 对")
        for source, target, error in errors[:10]:
            print(f"  {source} / {target}: {error}")

        raise SystemExit(
            "由于存在读取或检查错误，未保存结果表。"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "component",
        "source_image",
        "target_image",
        "direction",
        "overlap_pixels",
        "different_pixels",
        "status",
    ]

    with OUTPUT_PATH.open(
        "x", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    identical = sum(
        result["status"] == "identical"
        for result in results
    )
    different = [
        result for result in results
        if result["status"] == "different"
    ]

    print("\n检查完成。")
    print(f"逐像素一致：{identical} 对")
    print(f"存在像素差异：{len(different)} 对")
    print(f"结果表：{OUTPUT_PATH}")

    if different:
        print("\n前 10 对存在差异的切片：")
        for item in different[:10]:
            print(
                f"  组 {item['component']}："
                f"{item['source_image']} -> "
                f"{item['target_image']} "
                f"({item['direction']})，"
                f"不同像素 {item['different_pixels']:,}"
            )

    print(
        "\n注意：本次只检查 v2 各组内部，"
        "不能排除不同组之间仍有空间重叠。"
    )


if __name__ == "__main__":
    main()