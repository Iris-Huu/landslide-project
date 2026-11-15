"""
检查 Lushan_1 的 EPSG:4543 假设下，
其滑坡标签与 Lushan_3 标签是否有空间重合。

程序不修改原始 TIFF。标签覆盖范围使用虚拟 alpha 波段；
它表示源栅格的几何/元数据覆盖，不等于真实影像有效区。
"""

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "data/raw/GDCLD/Lushan_extracted"

IMAGE1 = ROOT / "test_data/Lushan/Lushan_1.tif"
LABEL1 = ROOT / "test_label/Lushan/Lushan_1.tif"
LABEL3 = ROOT / "test_label/Lushan/Lushan_3.tif"

OUTPUT = (
    PROJECT / "reports/tables/"
    "gdcld_lushan_crs_label_overlap_v1.json"
)

BLOCK = 2048


def main():
    if OUTPUT.exists():
        raise SystemExit(f"输出文件已存在，未覆盖：{OUTPUT}")

    counts = {
        "common_footprint_pixels": 0,
        "label1_positive_pixels": 0,
        "label3_positive_pixels": 0,
        "intersection_pixels": 0,
        "union_pixels": 0,
    }

    with rasterio.open(IMAGE1) as image1, \
         rasterio.open(LABEL1) as label1, \
         rasterio.open(LABEL3) as label3:

        if image1.crs is None or label1.crs is None or label3.crs is None:
            raise SystemExit("至少一个文件缺少 CRS，已停止。")

        if image1.crs != label1.crs:
            raise SystemExit("Lushan_1 影像和标签声明的 CRS 不一致。")

        if label1.count != 1 or label3.count != 1:
            raise SystemExit("标签不是单波段，需先检查。")

        # 第一份标签按原声明的 CRS 对齐至 Lushan_1 影像网格。
        with WarpedVRT(
            label1,
            crs=image1.crs,
            transform=image1.transform,
            width=image1.width,
            height=image1.height,
            resampling=Resampling.nearest,
            add_alpha=True,
        ) as aligned1, WarpedVRT(
            # 第二份标签映射至“假设为 4543”的 Lushan_1 网格。
            label3,
            crs="EPSG:4543",
            transform=image1.transform,
            width=image1.width,
            height=image1.height,
            resampling=Resampling.nearest,
            add_alpha=True,
        ) as aligned3:

            if aligned1.count != 2 or aligned3.count != 2:
                raise SystemExit("虚拟 alpha 波段结构异常，已停止。")

            total_blocks = (
                ((image1.height + BLOCK - 1) // BLOCK)
                * ((image1.width + BLOCK - 1) // BLOCK)
            )

            done = 0

            for row in range(0, image1.height, BLOCK):
                for col in range(0, image1.width, BLOCK):
                    width = min(BLOCK, image1.width - col)
                    height = min(BLOCK, image1.height - row)
                    window = Window(col, row, width, height)

                    first = aligned1.read(1, window=window)
                    third = aligned3.read(1, window=window)

                    valid1 = aligned1.read(2, window=window) > 0
                    valid3 = aligned3.read(2, window=window) > 0
                    common = valid1 & valid3

                    positive1 = (first == 1) & common
                    positive3 = (third == 1) & common

                    counts["common_footprint_pixels"] += int(common.sum())
                    counts["label1_positive_pixels"] += int(positive1.sum())
                    counts["label3_positive_pixels"] += int(positive3.sum())
                    counts["intersection_pixels"] += int(
                        (positive1 & positive3).sum()
                    )
                    counts["union_pixels"] += int(
                        (positive1 | positive3).sum()
                    )

                    done += 1
                    if done % 10 == 0 or done == total_blocks:
                        print(
                            f"已处理 {done}/{total_blocks} 个窗口",
                            flush=True,
                        )

    if counts["common_footprint_pixels"] == 0:
        raise SystemExit("没有检测到共同栅格覆盖，未保存结果。")

    union = counts["union_pixels"]
    counts["foreground_iou"] = (
        counts["intersection_pixels"] / union
        if union else None
    )

    report = {
        "status": "exploratory_hypothesis_not_confirmed",
        "hypothesis": (
            "Interpret Lushan_1 numeric grid as EPSG:4543 "
            "for this virtual comparison only"
        ),
        "resampling": "nearest",
        "counts": counts,
        "limitations": [
            "No original CRS or pixel data was modified.",
            "Alpha coverage is not a validated imagery-validity mask.",
            "Labels may share annotation sources and are not independent ground truth.",
            "Different resolutions and annotation choices affect overlap.",
        ],
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n结果：")
    print(f"  共同标签栅格覆盖：{counts['common_footprint_pixels']:,} 像素")
    print(f"  Lushan_1 滑坡像素：{counts['label1_positive_pixels']:,}")
    print(f"  Lushan_3 滑坡像素：{counts['label3_positive_pixels']:,}")
    print(f"  两者交集：{counts['intersection_pixels']:,}")
    print(f"  两者并集：{counts['union_pixels']:,}")

    if counts["foreground_iou"] is not None:
        print(f"  探索性前景 IoU：{counts['foreground_iou']:.4f}")
    else:
        print("  共同范围内没有前景，IoU 无法计算。")

    print(f"\n结果已保存：{OUTPUT}")
    print("注意：这不是 CRS 正确性的独立证明，也不是模型评价。")


if __name__ == "__main__":
    main()
