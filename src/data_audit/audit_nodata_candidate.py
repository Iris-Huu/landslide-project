"""
审计 Moxitaidi 影像的候选无数据区域。

候选规则：
    与图像边缘连通的纯黑或纯白像素。

重要限制：
    该规则已观察到会误选部分疑似高亮滑坡地表。
    结果仅供审计，不可直接作为最终训练或评价掩码。
"""

from pathlib import Path
import csv
import warnings

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from scipy import ndimage


# 脚本位于项目的 src/data_audit 下，向上两级找到项目根目录
PROJECT_DIR = Path(__file__).resolve().parents[2]
CAS_DIR = PROJECT_DIR / "data" / "raw" / "CAS"

OUTPUT_PATH = (
    PROJECT_DIR
    / "reports"
    / "tables"
    / "moxitaidi_nodata_audit_v1.csv"
)

SUBSETS = [
    "Moxitaidi_SAT",
    "Moxitaidi_UAV_0.6m",
]

warnings.filterwarnings(
    "ignore",
    category=NotGeoreferencedWarning,
)


def find_candidate(image):
    """
    找出与影像边缘连通的纯黑或纯白像素。

    输入 image 的形状为：通道 × 高 × 宽。
    返回的布尔数组中，True 表示候选区域。
    """
    pure_black = np.all(image == 0, axis=0)
    pure_white = np.all(image == 255, axis=0)
    possible = pure_black | pure_white

    # 在图像四条边设置搜索起点
    seeds = np.zeros(possible.shape, dtype=bool)
    seeds[0, :] = possible[0, :]
    seeds[-1, :] = possible[-1, :]
    seeds[:, 0] = possible[:, 0]
    seeds[:, -1] = possible[:, -1]

    # 沿上下左右方向传播，找出与边缘连通的候选像素
    return ndimage.binary_propagation(
        seeds,
        mask=possible,
    )


def audit_subset(subset):
    """逐张检查一个子集。"""
    root = CAS_DIR / subset
    image_dir = root / "img"
    mask_dir = root / "mask"

    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(
            f"{subset} 缺少 img 或 mask 目录：{root}"
        )

    images = sorted(image_dir.glob("*.tif"))

    if not images:
        raise RuntimeError(f"{subset} 没有找到 .tif 影像")

    results = []
    errors = []

    print(f"\n开始检查：{subset}，共 {len(images)} 张", flush=True)

    for index, image_path in enumerate(images, start=1):
        mask_path = mask_dir / image_path.name

        try:
            with rasterio.open(image_path) as src:
                image = src.read()

            with rasterio.open(mask_path) as src:
                mask = src.read(1)

            if image.shape[0] != 3:
                raise ValueError(
                    f"影像通道数为 {image.shape[0]}，预期为 3"
                )

            if image.shape[1:] != mask.shape:
                raise ValueError(
                    f"影像与掩码尺寸不一致："
                    f"{image.shape[1:]} / {mask.shape}"
                )

            if not np.isin(np.unique(mask), [0, 1]).all():
                raise ValueError("掩码含有 0 和 1 以外的值")

            candidate = find_candidate(image)
            landslide = mask == 1
            overlap = candidate & landslide

            results.append({
                "subset": subset,
                "filename": image_path.name,
                "total_pixels": int(candidate.size),
                "candidate_pixels": int(candidate.sum()),
                "candidate_percent": float(candidate.mean() * 100),
                "landslide_pixels": int(landslide.sum()),
                "overlap_pixels": int(overlap.sum()),
            })

        except Exception as error:
            errors.append((image_path.name, str(error)))

        if index % 200 == 0 or index == len(images):
            print(f"  已处理 {index}/{len(images)}", flush=True)

    return results, errors


def print_summary(subset, results, errors):
    """显示子集汇总结果。"""
    ratios = np.array([
        row["candidate_percent"] for row in results
    ])

    total_landslide = sum(
        row["landslide_pixels"] for row in results
    )
    total_overlap = sum(
        row["overlap_pixels"] for row in results
    )
    affected = sum(
        row["overlap_pixels"] > 0 for row in results
    )

    overlap_percent = (
        total_overlap / total_landslide * 100
        if total_landslide else 0
    )

    print(f"\n{subset} 汇总：")
    print(f"  成功检查：{len(results)} 张")
    print(f"  读取或检查失败：{len(errors)} 张")

    if len(ratios):
        print(f"  候选区域为 0：{np.sum(ratios == 0)} 张")
        print(f"  存在候选区域：{np.sum(ratios > 0)} 张")
        print(f"  平均候选区域占比：{ratios.mean():.2f}%")
        print(f"  最大候选区域占比：{ratios.max():.2f}%")

    print(f"  与滑坡标注有重叠：{affected} 张")
    print(f"  滑坡标注像素总数：{total_landslide:,}")
    print(f"  重叠像素总数：{total_overlap:,}")
    print(f"  重叠占滑坡标注：{overlap_percent:.4f}%")

    if errors:
        print("  前 10 条错误：")
        for filename, error in errors[:10]:
            print(f"    {filename}: {error}")


def main():
    if OUTPUT_PATH.exists():
        raise SystemExit(
            f"输出文件已存在，为避免覆盖而停止：{OUTPUT_PATH}"
        )

    all_results = []
    all_errors = []

    for subset in SUBSETS:
        results, errors = audit_subset(subset)
        all_results.extend(results)
        all_errors.extend(
            (subset, filename, error)
            for filename, error in errors
        )
        print_summary(subset, results, errors)

    # 有错误时不生成看似完整的结果表，先检查错误
    if all_errors:
        raise SystemExit(
            f"\n共有 {len(all_errors)} 张影像未完成检查，"
            "未保存结果表。请先查看上方错误。"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "subset",
        "filename",
        "total_pixels",
        "candidate_pixels",
        "candidate_percent",
        "landslide_pixels",
        "overlap_pixels",
    ]

    # x 模式确保已有文件不会被覆盖
    with OUTPUT_PATH.open(
        "x", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n逐张结果已保存：{OUTPUT_PATH}")
    print("注意：候选区域存在已知误判，不是最终无数据掩码。")


if __name__ == "__main__":
    main()