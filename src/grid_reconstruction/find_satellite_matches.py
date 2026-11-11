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
    main()