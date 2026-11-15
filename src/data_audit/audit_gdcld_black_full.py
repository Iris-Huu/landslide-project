"""
分窗口清点 GDCLD Lushan_1/2 整景中的纯黑 RGB 像素。

标签使用最近邻虚拟对齐到影像网格。
纯黑不等于已确认的无数据区；程序不生成有效区掩码。
"""

import csv
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "data/raw/GDCLD/Lushan_extracted"
OUTPUT = (
    PROJECT / "reports/tables/"
    "gdcld_lushan_black_full_v1.csv"
)

SCENES = ["Lushan_1.tif", "Lushan_2.tif"]
BLOCK = 1024


def audit_scene(name):
    image_path = ROOT / "test_data/Lushan" / name
    label_path = ROOT / "test_label/Lushan" / name

    stats = {
        "scene": name,
        "image_pixels": 0,
        "pure_black_pixels": 0,
        "image_mask_invalid_pixels": 0,
        "aligned_label_mask_valid_pixels": 0,
        "aligned_landslide_pixels": 0,
        "black_and_aligned_landslide_pixels": 0,
    }

    with rasterio.open(image_path) as image, rasterio.open(label_path) as label:
        if image.crs is None or label.crs is None:
            raise ValueError(f"{name} 缺少坐标系")

        if image.crs != label.crs:
            raise ValueError(f"{name} 影像与标签坐标系不同")

        if image.count < 3:
            raise ValueError(f"{name} 影像不足三个波段")

        total_blocks = (
            ((image.height + BLOCK - 1) // BLOCK)
            * ((image.width + BLOCK - 1) // BLOCK)
        )

        with WarpedVRT(
            label,
            crs=image.crs,
            transform=image.transform,
            width=image.width,
            height=image.height,
            resampling=Resampling.nearest,
        ) as aligned:

            count = 0

            for row in range(0, image.height, BLOCK):
                for col in range(0, image.width, BLOCK):
                    width = min(BLOCK, image.width - col)
                    height = min(BLOCK, image.height - row)
                    window = Window(col, row, width, height)

                    rgb = image.read([1, 2, 3], window=window)
                    image_masks = image.read_masks(
                        [1, 2, 3], window=window
                    )

                    label_array = aligned.read(1, window=window)
                    label_valid = (
                        aligned.read_masks(1, window=window) > 0
                    )

                    black = np.all(rgb == 0, axis=0)
                    image_valid = np.all(image_masks > 0, axis=0)
                    landslide = (label_array == 1) & label_valid

                    stats["image_pixels"] += int(black.size)
                    stats["pure_black_pixels"] += int(black.sum())
                    stats["image_mask_invalid_pixels"] += int(
                        (~image_valid).sum()
                    )
                    stats["aligned_label_mask_valid_pixels"] += int(
                        label_valid.sum()
                    )
                    stats["aligned_landslide_pixels"] += int(
                        landslide.sum()
                    )
                    stats["black_and_aligned_landslide_pixels"] += int(
                        (black & landslide).sum()
                    )

                    count += 1

                    if count % 50 == 0 or count == total_blocks:
                        print(
                            f"  {name}: {count}/{total_blocks} 个窗口",
                            flush=True,
                        )

    return stats


def main():
    if OUTPUT.exists():
        raise SystemExit(f"输出文件已存在，未覆盖：{OUTPUT}")

    results = []

    for name in SCENES:
        print(f"\n开始整景清点：{name}", flush=True)
        stats = audit_scene(name)
        results.append(stats)

        total = stats["image_pixels"]
        black = stats["pure_black_pixels"]
        landslide = stats["aligned_landslide_pixels"]
        overlap = stats["black_and_aligned_landslide_pixels"]

        print(f"  影像总像素：{total:,}")
        print(f"  纯黑像素：{black:,}（{black / total * 100:.2f}%）")
        print(
            f"  影像自带掩码标记为无效："
            f"{stats['image_mask_invalid_pixels']:,}"
        )
        print(f"  对齐后滑坡标注像素：{landslide:,}")
        print(f"  纯黑且为滑坡标注：{overlap:,}")

        if landslide:
            print(
                f"  上述重叠占滑坡标注："
                f"{overlap / landslide * 100:.4f}%"
            )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    print(f"\n结果已保存：{OUTPUT}")
    print("注意：纯黑像素不是已确认的无数据区。")
    print("标签有效性掩码依赖源文件元数据。")


if __name__ == "__main__":
    main()
