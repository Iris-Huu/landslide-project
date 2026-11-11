"""
寻找 Moxitaidi UAV 切片的 50% 对角重叠候选。

检查两个方向：
1. 右下：源图右下角 = 目标图左上角
2. 左下：源图左下角 = 目标图右上角

注意：
- 输出仅是候选关系，不能直接作为最终网格。
- 纯黑、纯白或其他重复内容可能产生假匹配。
- 纯黑或纯白像素也可能是真实地物，不直接剔除。
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
    / "Moxitaidi_SAT"
    / "img"
)

OUTPUT_PATH = (
    PROJECT_DIR
    / "data"
    / "interim"
    / "reconstructed_grids"
    /"satellite_diagonal_candidates.csv"
)

warnings.filterwarnings(
    "ignore",
    category=NotGeoreferencedWarning,
)


def image_hash(array):
    """计算影像区域的 SHA-256 指纹。"""
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def read_image(path):
    """读取一张 RGB 影像，并检查尺寸。"""
    with rasterio.open(path) as src:
        image = src.read()

    if image.shape != (3, 512, 512):
        raise ValueError(
            f"{path.name} 形状为 {image.shape}，"
            "预期为 (3, 512, 512)"
        )

    return image


def pure_extreme_percent(region):
    """计算角部中纯黑或纯白像素所占的百分比。"""
    black = np.all(region == 0, axis=0)
    white = np.all(region == 255, axis=0)
    return float(np.mean(black | white) * 100)


def main():
    if OUTPUT_PATH.exists():
        raise SystemExit(
            f"输出文件已存在，为避免覆盖而停止：{OUTPUT_PATH}"
        )

    images = sorted(IMAGE_DIR.glob("*.tif"))

    if not images:
        raise SystemExit(f"未找到 TIFF 影像：{IMAGE_DIR}")

    # 目标图左上角的指纹索引：供“右下”匹配使用
    top_left_index = defaultdict(list)

    # 目标图右上角的指纹索引：供“左下”匹配使用
    top_right_index = defaultdict(list)

    # 每张源图的右下角和左下角指纹
    source_hashes = {}

    print(
        f"开始读取 {len(images)} 张影像并计算角部指纹。",
        flush=True,
    )

    for index, path in enumerate(images, start=1):
        image = read_image(path)

        top_left = image[:, :256, :256]
        top_right = image[:, :256, 256:]
        bottom_left = image[:, 256:, :256]
        bottom_right = image[:, 256:, 256:]

        top_left_index[image_hash(top_left)].append(path)
        top_right_index[image_hash(top_right)].append(path)

        source_hashes[path] = {
            "down_right": image_hash(bottom_right),
            "down_left": image_hash(bottom_left),
        }

        if index % 200 == 0 or index == len(images):
            print(f"  已处理 {index}/{len(images)}", flush=True)

    pairs = []

    for source_path in images:
        source = read_image(source_path)

        directions = [
            (
                "down_right",
                source[:, 256:, 256:],
                top_left_index,
                lambda image: image[:, :256, :256],
            ),
            (
                "down_left",
                source[:, 256:, :256],
                top_right_index,
                lambda image: image[:, :256, 256:],
            ),
        ]

        for direction, source_corner, index_map, get_target_corner in directions:
            digest = source_hashes[source_path][direction]
            candidates = index_map.get(digest, [])

            for target_path in candidates:
                if target_path == source_path:
                    continue

                target = read_image(target_path)
                target_corner = get_target_corner(target)

                if np.array_equal(source_corner, target_corner):
                    pairs.append({
                        "source_image": source_path.name,
                        "target_image": target_path.name,
                        "direction": direction,
                        "pure_black_white_percent": (
                            pure_extreme_percent(source_corner)
                        ),
                    })

    # 分别统计每个方向上的源、目标候选数量
    source_counts = defaultdict(int)
    target_counts = defaultdict(int)

    for pair in pairs:
        direction = pair["direction"]
        source_counts[(direction, pair["source_image"])] += 1
        target_counts[(direction, pair["target_image"])] += 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "source_image",
        "target_image",
        "direction",
        "source_direction_candidates",
        "target_direction_candidates",
        "pure_black_white_percent",
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
    print(f"对角候选影像对总数：{len(pairs)}")

    for direction in ["down_right", "down_left"]:
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
        print(f"  候选对：{len(selected)}")
        print(f"  双向唯一候选：{unique_count}")
        print(f"  存在方向歧义的候选：{len(selected) - unique_count}")

    print(f"\n结果已保存：{OUTPUT_PATH}")

    print("\n前 5 组候选：")
    for pair in pairs[:5]:
        print(
            f"  {pair['source_image']} "
            f"-> {pair['target_image']} "
            f"({pair['direction']})"
        )

    print(
        "\n注意：结果尚需结合已有水平、垂直网格"
        "检查一致性；不要直接作为最终坐标。"
    )


if __name__ == "__main__":
    main()