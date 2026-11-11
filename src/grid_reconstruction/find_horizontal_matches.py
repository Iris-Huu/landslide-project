"""
查找 Moxitaidi UAV 切片中 50% 水平重叠的候选影像对。

假设：
    左图右半部分与右图左半部分逐像素相同。

注意：
    结果仅是候选关系。重复内容可能产生歧义，
    不能直接把所有候选关系当作最终网格。
"""

from collections import defaultdict
from pathlib import Path
import csv
import hashlib
import warnings

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning


PROJECT_DIR = Path(__file__).resolve().parents[2]

IMAGE_DIR = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "CAS"
    / "Moxitaidi_UAV_0.6m"
    / "img"
)

OUTPUT_PATH = (
    PROJECT_DIR
    / "data"
    / "interim"
    / "reconstructed_grids"
    / "uav_horizontal_candidates.csv"
)

warnings.filterwarnings(
    "ignore",
    category=NotGeoreferencedWarning,
)


def image_hash(array):
    """计算图像区域的 SHA-256 指纹。"""
    return hashlib.sha256(
        np.ascontiguousarray(array).tobytes()
    ).hexdigest()


def read_image(path):
    """读取并检查一张 RGB 切片。"""
    with rasterio.open(path) as src:
        image = src.read()

    if image.shape != (3, 512, 512):
        raise ValueError(
            f"{path.name} 的形状为 {image.shape}，"
            "预期为 (3, 512, 512)"
        )

    return image


def main():
    if OUTPUT_PATH.exists():
        raise SystemExit(
            f"输出文件已存在，为避免覆盖而停止：{OUTPUT_PATH}"
        )

    images = sorted(IMAGE_DIR.glob("*.tif"))
    if not images:
        raise SystemExit(f"未找到 TIFF 影像：{IMAGE_DIR}")

    # 左半幅指纹 -> 可能位于右侧的文件
    left_half_index = defaultdict(list)

    # 保存每张影像右半幅的指纹
    right_half_hashes = {}

    print(f"开始读取 {len(images)} 张影像并计算指纹。", flush=True)

    for index, image_path in enumerate(images, start=1):
        image = read_image(image_path)

        left_half = image[:, :, :256]
        right_half = image[:, :, 256:]

        left_half_index[image_hash(left_half)].append(image_path)
        right_half_hashes[image_path] = image_hash(right_half)

        if index % 200 == 0 or index == len(images):
            print(f"  已处理 {index}/{len(images)}", flush=True)

    # 根据相同指纹寻找候选，并再次逐像素核对
    pairs = []

    for index, left_path in enumerate(images, start=1):
        matches = left_half_index.get(
            right_half_hashes[left_path],
            [],
        )

        # 自身不作为邻居
        matches = [path for path in matches if path != left_path]

        if not matches:
            continue

        left_image = read_image(left_path)
        left_overlap = left_image[:, :, 256:]

        for right_path in matches:
            right_image = read_image(right_path)
            right_overlap = right_image[:, :, :256]

            if np.array_equal(left_overlap, right_overlap):
                pairs.append((left_path.name, right_path.name))

    # 计算每个方向上的候选数量，标记多对一或一对多
    right_counts = defaultdict(int)
    left_counts = defaultdict(int)

    for left_name, right_name in pairs:
        right_counts[left_name] += 1
        left_counts[right_name] += 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "left_image",
        "right_image",
        "left_image_right_candidates",
        "right_image_left_candidates",
        "status",
    ]

    with OUTPUT_PATH.open(
        "x", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for left_name, right_name in pairs:
            right_count = right_counts[left_name]
            left_count = left_counts[right_name]

            status = (
                "unique_candidate"
                if right_count == 1 and left_count == 1
                else "ambiguous"
            )

            writer.writerow({
                "left_image": left_name,
                "right_image": right_name,
                "left_image_right_candidates": right_count,
                "right_image_left_candidates": left_count,
                "status": status,
            })

    unique_count = sum(
        right_counts[left_name] == 1
        and left_counts[right_name] == 1
        for left_name, right_name in pairs
    )

    print("\n检查完成。")
    print(f"影像总数：{len(images)}")
    print(f"逐像素相同的候选影像对：{len(pairs)}")
    print(f"双向唯一候选：{unique_count}")
    print(f"存在方向歧义的候选：{len(pairs) - unique_count}")
    print(f"结果已保存：{OUTPUT_PATH}")

    print("\n前 5 组候选：")
    for left_name, right_name in pairs[:5]:
        print(f"  {left_name} -> {right_name}")

    print(
        "\n注意：即使是双向唯一候选，也需要进一步"
        "检查图像内容和网格整体一致性。"
    )


if __name__ == "__main__":
    main()