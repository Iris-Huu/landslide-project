"""
查找 Moxitaidi 卫星切片中 50% 重叠的候选邻居。

检查方向：
- right：源图右半边 = 目标图左半边；
- down：源图下半边 = 目标图上半边。

结果仅为候选关系。重复内容可能产生歧义，
即使方向唯一，也需要进一步核查。
"""

from collections import Counter, defaultdict
from pathlib import Path
import csv
import hashlib
import warnings

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning


PROJECT = Path(__file__).resolve().parents[2]

IMAGE_DIR = (
    PROJECT
    / "data"
    / "raw"
    / "CAS"
    / "Moxitaidi_SAT"
    / "img"
)

OUTPUT_PATH = (
    PROJECT
    / "data"
    / "interim"
    / "reconstructed_grids"
    / "satellite_horizontal_vertical_candidates.csv"
)

warnings.filterwarnings(
    "ignore",
    category=NotGeoreferencedWarning,
)


def read_image(path):
    """读取并检查一张 RGB 影像。"""
    with rasterio.open(path) as src:
        image = src.read()

    if image.shape != (3, 512, 512):
        raise ValueError(
            f"{path.name} 的形状为 {image.shape}，"
            "预期为 (3, 512, 512)"
        )

    return image


def image_hash(region):
    """计算图像区域的 SHA-256 指纹。"""
    data = np.ascontiguousarray(region).tobytes()
    return hashlib.sha256(data).hexdigest()


def main():
    if OUTPUT_PATH.exists():
        raise SystemExit(
            f"输出文件已存在，为避免覆盖而停止：{OUTPUT_PATH}"
        )

    images = sorted(IMAGE_DIR.glob("*.tif"))

    if not images:
        raise SystemExit(f"未找到 TIFF 影像：{IMAGE_DIR}")

    # 目标图左半边和上半边的索引
    left_index = defaultdict(list)
    top_index = defaultdict(list)

    # 保存源图右半边、下半边的指纹
    source_hashes = {}

    print(
        f"开始读取 {len(images)} 张卫星影像并计算指纹。",
        flush=True,
    )

    for index, path in enumerate(images, start=1):
        image = read_image(path)

        left_index[image_hash(image[:, :, :256])].append(path)
        top_index[image_hash(image[:, :256, :])].append(path)

        source_hashes[path] = {
            "right": image_hash(image[:, :, 256:]),
            "down": image_hash(image[:, 256:, :]),
        }

        if index % 200 == 0 or index == len(images):
            print(f"  已计算 {index}/{len(images)} 张", flush=True)

    pairs = []

    print("\n开始核对候选区域的像素。", flush=True)

    for index, source_path in enumerate(images, start=1):
        source = read_image(source_path)

        # 右方候选：源图右半边与目标图左半边
        right_candidates = left_index.get(
            source_hashes[source_path]["right"], []
        )

        for target_path in right_candidates:
            if target_path == source_path:
                continue

            target = read_image(target_path)

            if np.array_equal(
                source[:, :, 256:],
                target[:, :, :256],
            ):
                pairs.append({
                    "source_image": source_path.name,
                    "target_image": target_path.name,
                    "direction": "right",
                })

        # 下方候选：源图下半边与目标图上半边
        down_candidates = top_index.get(
            source_hashes[source_path]["down"], []
        )

        for target_path in down_candidates:
            if target_path == source_path:
                continue

            target = read_image(target_path)

            if np.array_equal(
                source[:, 256:, :],
                target[:, :256, :],
            ):
                pairs.append({
                    "source_image": source_path.name,
                    "target_image": target_path.name,
                    "direction": "down",
                })

        if index % 200 == 0 or index == len(images):
            print(f"  已核对 {index}/{len(images)} 张", flush=True)

    # 分别统计每个方向的候选数量
    source_counts = Counter()
    target_counts = Counter()

    for pair in pairs:
        direction = pair["direction"]
        source_counts[(direction, pair["source_image"])] += 1
        target_counts[(direction, pair["target_image"])] += 1

    fieldnames = [
        "source_image",
        "target_image",
        "direction",
        "source_direction_candidates",
        "target_direction_candidates",
        "status",
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open(
        "x", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for pair in pairs:
            direction = pair["direction"]
            source_name = pair["source_image"]
            target_name = pair["target_image"]

            source_count = source_counts[(direction, source_name)]
            target_count = target_counts[(direction, target_name)]

            status = (
                "unique_candidate"
                if source_count == 1 and target_count == 1
                else "ambiguous"
            )

            writer.writerow({
                **pair,
                "source_direction_candidates": source_count,
                "target_direction_candidates": target_count,
                "status": status,
            })

    print("\n检查完成。")
    print(f"影像总数：{len(images)}")

    for direction in ["right", "down"]:
        selected = [
            pair for pair in pairs
            if pair["direction"] == direction
        ]

        unique_count = sum(
            source_counts[(direction, pair["source_image"])] == 1
            and target_counts[(direction, pair["target_image"])] == 1
            for pair in selected
        )

        print(f"\n方向：{direction}")
        print(f"  逐像素一致的候选对：{len(selected)}")
        print(f"  方向上唯一的候选：{unique_count}")
        print(f"  有方向歧义的候选：{len(selected) - unique_count}")

    print(f"\n结果已保存：{OUTPUT_PATH}")

    print("\n前 5 条候选：")
    for pair in pairs[:5]:
        print(
            f"  {pair['source_image']} -> "
            f"{pair['target_image']} "
            f"({pair['direction']})"
        )

    print(
        "\n注意：候选关系还需要目视检查和网格一致性验证。"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="查找 Moxitaidi 水平和垂直重叠候选。"
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["uav", "satellite"],
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--legacy-output-dir",
        type=Path,
        help="可选：将无人机候选另存为旧网格脚本使用的两张表",
    )

    args = parser.parse_args()

    if args.legacy_output_dir is not None:
        if args.dataset != "uav":
            parser.error("--legacy-output-dir 只用于无人机数据")

        horizontal_path = (
            args.legacy_output_dir / "uav_horizontal_candidates.csv"
        )
        vertical_path = (
            args.legacy_output_dir / "uav_vertical_candidates.csv"
        )

        outputs = [args.output, horizontal_path, vertical_path]

        if len({path.resolve() for path in outputs}) != len(outputs):
            parser.error("合并表与两张旧格式表必须使用不同路径")

        for path in outputs:
            if path.exists():
                parser.error(f"输出文件已存在，不会覆盖：{path}")

    subset = (
        "Moxitaidi_UAV_0.6m"
        if args.dataset == "uav"
        else "Moxitaidi_SAT"
    )

    IMAGE_DIR = (
        PROJECT / "data" / "raw" / "CAS" / subset / "img"
    )

    OUTPUT_PATH = args.output

    main()

    if args.legacy_output_dir is not None:
        with OUTPUT_PATH.open(
            encoding="utf-8", newline=""
        ) as file:
            rows = list(csv.DictReader(file))

        horizontal_fields = [
            "left_image",
            "right_image",
            "left_image_right_candidates",
            "right_image_left_candidates",
            "status",
        ]

        vertical_fields = [
            "top_image",
            "bottom_image",
            "top_image_bottom_candidates",
            "bottom_image_top_candidates",
            "status",
        ]

        args.legacy_output_dir.mkdir(parents=True, exist_ok=True)

        with horizontal_path.open(
            "x", encoding="utf-8", newline=""
        ) as file:
            writer = csv.DictWriter(file, fieldnames=horizontal_fields)
            writer.writeheader()

            for row in rows:
                if row["direction"] == "right":
                    writer.writerow({
                        "left_image": row["source_image"],
                        "right_image": row["target_image"],
                        "left_image_right_candidates":
                            row["source_direction_candidates"],
                        "right_image_left_candidates":
                            row["target_direction_candidates"],
                        "status": row["status"],
                    })

        with vertical_path.open(
            "x", encoding="utf-8", newline=""
        ) as file:
            writer = csv.DictWriter(file, fieldnames=vertical_fields)
            writer.writeheader()

            for row in rows:
                if row["direction"] == "down":
                    writer.writerow({
                        "top_image": row["source_image"],
                        "bottom_image": row["target_image"],
                        "top_image_bottom_candidates":
                            row["source_direction_candidates"],
                        "bottom_image_top_candidates":
                            row["target_direction_candidates"],
                        "status": row["status"],
                    })

        print(f"无人机水平旧格式表：{horizontal_path}")
        print(f"无人机垂直旧格式表：{vertical_path}")