"""
复用现有水平/垂直匹配程序，检查三个异址子集。

先按区域指纹估计候选对数量；若组合过多则停止，
避免重复内容导致高内存或长时间计算。
"""

import argparse
from collections import Counter
from pathlib import Path

import rasterio

import find_horizontal_vertical_matches as matcher


PROJECT = Path(__file__).resolve().parents[2]
CAS = PROJECT / "data/raw/CAS"

DATASETS = {
    "longxi_sat": "Longxi_SAT",
    "longxi_uav": "Longxi_UAV",
    "jiuzhai_uav_05": "Jiuzhai_UAV_0.5m",
}

MAX_CANDIDATES = 50_000


def estimate_candidates(images):
    """
    根据 SHA-256 指纹统计可能的匹配对数。

    此处只统计，不把指纹相同直接作为最终匹配；
    正式匹配仍由原程序逐像素核对。
    """
    targets = {
        "right": Counter(),
        "down": Counter(),
    }
    sources = {
        "right": Counter(),
        "down": Counter(),
    }
    self_matches = Counter()

    for index, path in enumerate(images, start=1):
        image = matcher.read_image(path)

        left = matcher.image_hash(image[:, :, :256])
        right = matcher.image_hash(image[:, :, 256:])
        top = matcher.image_hash(image[:, :256, :])
        bottom = matcher.image_hash(image[:, 256:, :])

        targets["right"][left] += 1
        sources["right"][right] += 1

        targets["down"][top] += 1
        sources["down"][bottom] += 1

        self_matches["right"] += int(right == left)
        self_matches["down"] += int(bottom == top)

        if index % 200 == 0 or index == len(images):
            print(
                f"  预检查 {index}/{len(images)} 张",
                flush=True,
            )

    estimates = {}

    for direction in ["right", "down"]:
        possible = sum(
            count * targets[direction].get(digest, 0)
            for digest, count in sources[direction].items()
        )

        estimates[direction] = possible - self_matches[direction]

    return estimates


def main():
    parser = argparse.ArgumentParser(
        description="检查异址子集的水平和垂直重叠候选"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(DATASETS),
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"输出已存在，未覆盖：{args.output}")

    image_dir = CAS / DATASETS[args.dataset] / "img"

    if not image_dir.is_dir():
        raise SystemExit(f"影像目录不存在：{image_dir}")

    images = sorted(image_dir.glob("*.tif"))

    if not images:
        raise SystemExit(f"未找到 .tif 影像：{image_dir}")

    print(f"数据集：{args.dataset}")
    print(f"影像数：{len(images)}")
    print("开始估计候选组合数量……", flush=True)

    # 限制 GDAL 缓存；原程序仍逐张读取图像。
    with rasterio.Env(GDAL_CACHEMAX=32 * 1024 * 1024):
        estimates = estimate_candidates(images)

        print("\n指纹一致的可能候选对：")
        for direction, count in estimates.items():
            print(f"  {direction}: {count}")

        total = sum(estimates.values())
        print(f"  合计：{total}")

        if total > MAX_CANDIDATES:
            raise SystemExit(
                f"候选对超过 {MAX_CANDIDATES:,}，"
                "为避免资源消耗过大已停止；请检查重复区域。"
            )

        # 复用已验证程序，修改其本次输入和输出位置。
        matcher.IMAGE_DIR = image_dir
        matcher.OUTPUT_PATH = args.output

        print("\n开始逐像素核对候选……", flush=True)
        matcher.main()

    print(
        "\n注意：原匹配程序的部分打印文字可能写着“卫星影像”；"
        "实际输入以上方数据集名称为准。"
    )
    print("结果仅是水平/垂直候选，还不是最终空间划分。")


if __name__ == "__main__":
    main()
