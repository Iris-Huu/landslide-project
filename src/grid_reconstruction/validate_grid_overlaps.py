"""
通用网格重叠验证。

用途：
- 检查同一连通组中，按网格坐标应当重叠的影像；
- 支持 UAV 或卫星数据；
- 可与先前保存的验证 CSV 逐对比较。

注意：
- 只检查同一连通组内部；
- 不排除不同连通组之间存在重叠；
- 不修改原始影像或已有结果。
"""

from pathlib import Path
import argparse
import csv
import warnings

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning


warnings.filterwarnings(
    "ignore",
    category=NotGeoreferencedWarning,
)


def read_image(folder, filename):
    path = folder / filename

    with rasterio.open(path) as src:
        image = src.read()

    if image.shape != (3, 512, 512):
        raise ValueError(
            f"{filename}: 形状异常 {image.shape}"
        )

    return image


def get_overlap(source, target, dr, dc):
    """
    根据目标影像相对源影像的网格位移，
    提取对应的重叠区域。
    """
    dy = dr * 256
    dx = dc * 256

    source_part = source[
        :,
        max(dy, 0):min(512 + dy, 512),
        max(dx, 0):min(512 + dx, 512),
    ]

    target_part = target[
        :,
        max(-dy, 0):min(512 - dy, 512),
        max(-dx, 0):min(512 - dx, 512),
    ]

    return source_part, target_part


def read_coordinates(path):
    positions = {}
    filenames = set()

    with path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            filename = row["filename"]

            key = (
                row["component"],
                int(row["grid_row"]),
                int(row["grid_col"]),
            )

            if filename in filenames:
                raise ValueError(f"坐标表出现重复影像：{filename}")

            if key in positions:
                raise ValueError(f"坐标表出现重复位置：{key}")

            if Path(filename).name != filename:
                raise ValueError(f"坐标表中的文件名异常：{filename}")

            positions[key] = filename
            filenames.add(filename)

    if not positions:
        raise ValueError("坐标表为空。")

    return positions


def read_previous_results(path):
    """读取旧程序产生的验证结果，用于逐对对照。"""
    previous = {}

    with path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            key = (
                row["component"],
                row["source_image"],
                row["target_image"],
                row["direction"],
            )

            if key in previous:
                raise ValueError(f"旧结果中存在重复记录：{key}")

            previous[key] = (
                int(row["overlap_pixels"]),
                int(row["different_pixels"]),
                row["status"],
            )

    return previous


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--images",
        required=True,
        type=Path,
        help="影像 img 目录",
    )

    parser.add_argument(
        "--coordinates",
        required=True,
        type=Path,
        help="网格坐标 CSV",
    )

    parser.add_argument(
        "--previous",
        required=True,
        type=Path,
        help="旧程序生成的重叠验证 CSV",
    )

    args = parser.parse_args()

    positions = read_coordinates(args.coordinates)
    previous = read_previous_results(args.previous)

    directions = [
        ("right", 0, 1),
        ("down", 1, 0),
        ("down_right", 1, 1),
        ("down_left", 1, -1),
    ]

    pairs = []

    for (component, row, col), source in positions.items():
        for direction, dr, dc in directions:
            target = positions.get(
                (component, row + dr, col + dc)
            )

            if target is not None:
                pairs.append(
                    (component, source, target, direction, dr, dc)
                )

    if not pairs:
        raise SystemExit(
            "没有找到组内重叠切片对，不能据此判断验证通过。"
        )

    print(f"坐标表切片：{len(positions)} 张")
    print(f"需要验证的组内重叠对：{len(pairs)} 对", flush=True)

    current = {}
    image_errors = []

    for index, pair in enumerate(pairs, start=1):
        component, source_name, target_name, direction, dr, dc = pair

        key = (
            component,
            source_name,
            target_name,
            direction,
        )

        try:
            source = read_image(args.images, source_name)
            target = read_image(args.images, target_name)

            source_part, target_part = get_overlap(
                source, target, dr, dc
            )

            if (
                source_part.shape != target_part.shape
                or source_part.size == 0
            ):
                raise ValueError("重叠区域尺寸异常")

            # 三个通道中任一个不同，即该像素位置不同
            different = np.any(
                source_part != target_part,
                axis=0,
            )

            different_pixels = int(different.sum())
            status = (
                "identical"
                if different_pixels == 0
                else "different"
            )

            current[key] = (
                int(different.size),
                different_pixels,
                status,
            )

        except Exception as error:
            image_errors.append((key, str(error)))

        if index % 500 == 0 or index == len(pairs):
            print(f"  已处理 {index}/{len(pairs)} 对", flush=True)

    missing_from_current = set(previous) - set(current)
    missing_from_previous = set(current) - set(previous)

    differences = [
        key for key in set(current) & set(previous)
        if current[key] != previous[key]
    ]

    print("\n与旧结果逐对比较：")
    print(f"  本次读取失败：{len(image_errors)} 对")
    print(f"  旧表有、本次没有：{len(missing_from_current)} 对")
    print(f"  本次有、旧表没有：{len(missing_from_previous)} 对")
    print(f"  像素统计或状态不一致：{len(differences)} 对")

    if image_errors:
        print("\n前 5 条读取错误：")
        for key, error in image_errors[:5]:
            print(f"  {key}: {error}")

    if differences:
        print("\n前 5 条结果差异：")
        for key in sorted(differences)[:5]:
            print(
                f"  {key}: "
                f"旧结果={previous[key]}；新结果={current[key]}"
            )

    if (
        image_errors
        or missing_from_current
        or missing_from_previous
        or differences
    ):
        raise SystemExit(
            "\n核对未通过：请保留旧脚本并检查上述差异。"
        )

    print("\n核对通过：新程序与旧验证表逐对一致。")
    print("注意：这不代表不同连通组之间没有重叠。")


if __name__ == "__main__":
    main()