"""
查找 Moxitaidi UAV 切片中 50% 垂直重叠的候选影像对。

假设：
    上图的下半部分与下图的上半部分逐像素相同。

结果仅是候选关系，不能直接当作最终网格。
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
    / "uav_vertical_candidates.csv"
)

warnings.filterwarnings(
    "ignore",
    category=NotGeoreferencedWarning,
)


def image_hash(array):
    """计算影像区域的 SHA-256 指纹。"""
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

    # 上半幅指纹 -> 可能位于下方的影像
    top_half_index = defaultdict(list)

    # 每张影像下半幅的指纹
    bottom_half_hashes = {}

    print(
        f"开始读取 {len(images)} 张影像并计算指纹。",
        flush=True,
    )

    for index, image_path in enumerate(images, start=1):
        image = read_image(image_path)

        top_half = image[:, :256, :]
        bottom_half = image[:, 256:, :]

        top_half_index[image_hash(top_half)].append(image_path)
        bottom_half_hashes[image_path] = image_hash(bottom_half)

        if index % 200 == 0 or index == len(images):
            print(f"  已处理 {index}/{len(images)}", flush=True)

    # 根据指纹寻找候选，并逐像素核对
    pairs = []

    for top_path in images:
        matches = top_half_index.get(
            bottom_half_hashes[top_path],
            [],
        )

        # 同一张影像不作为自身的邻居
        matches = [
            path for path in matches
            if path != top_path
        ]

        if not matches:
            continue

        top_image = read_image(top_path)
        top_overlap = top_image[:, 256:, :]

        for bottom_path in matches:
            bottom_image = read_image(bottom_path)
            bottom_overlap = bottom_image[:, :256, :]

            if np.array_equal(top_overlap, bottom_overlap):
                pairs.append(
                    (top_path.name, bottom_path.name)
                )

    # 统计每个方向上的候选数量
    bottom_counts = defaultdict(int)
    top_counts = defaultdict(int)

    for top_name, bottom_name in pairs:
        bottom_counts[top_name] += 1
        top_counts[bottom_name] += 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "top_image",
        "bottom_image",
        "top_image_bottom_candidates",
        "bottom_image_top_candidates",
        "status",
    ]

    with OUTPUT_PATH.open(
        "x",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for top_name, bottom_name in pairs:
            bottom_count = bottom_counts[top_name]
            top_count = top_counts[bottom_name]

            status = (
                "unique_candidate"
                if bottom_count == 1 and top_count == 1
                else "ambiguous"
            )

            writer.writerow({
                "top_image": top_name,
                "bottom_image": bottom_name,
                "top_image_bottom_candidates": bottom_count,
                "bottom_image_top_candidates": top_count,
                "status": status,
            })

    unique_count = sum(
        bottom_counts[top_name] == 1
        and top_counts[bottom_name] == 1
        for top_name, bottom_name in pairs
    )

    print("\n检查完成。")
    print(f"影像总数：{len(images)}")
    print(f"逐像素相同的候选影像对：{len(pairs)}")
    print(f"双向唯一候选：{unique_count}")
    print(f"存在方向歧义的候选：{len(pairs) - unique_count}")
    print(f"结果已保存：{OUTPUT_PATH}")

    print("\n前 5 组候选：")
    for top_name, bottom_name in pairs[:5]:
        print(f"  {top_name} -> {bottom_name}")

    print(
        "\n注意：双向唯一不代表已经确认最终邻接关系；"
        "还需检查图像内容与网格整体一致性。"
    )


if __name__ == "__main__":
    main()