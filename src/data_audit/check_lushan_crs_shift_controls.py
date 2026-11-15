"""
在 EPSG:4543 假设下，用固定共同覆盖范围比较标签及平移对照。

只在虚拟网格上读取，不修改原始 TIFF。
平移结果是探索性对照，不是 CRS 正确性的独立证明。
"""

import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
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
    "gdcld_lushan_crs_shift_controls_v1.json"
)

BLOCK = 1024

# (列方向位移, 行方向位移)，单位为 Lushan_1 网格像素。
SHIFTS = {
    "unshifted": (0, 0),
    "col_plus_512": (512, 0),
    "col_minus_512": (-512, 0),
    "row_plus_512": (0, 512),
    "row_minus_512": (0, -512),
}


def main():
    if OUTPUT.exists():
        raise SystemExit(f"结果文件已存在，未覆盖：{OUTPUT}")

    totals = {
        name: {"intersection": 0, "union": 0, "label3_positive": 0}
        for name in SHIFTS
    }
    common_pixels = 0
    label1_positive = 0

    with ExitStack() as stack:
        image1 = stack.enter_context(rasterio.open(IMAGE1))
        label1 = stack.enter_context(rasterio.open(LABEL1))
        label3 = stack.enter_context(rasterio.open(LABEL3))

        if image1.crs is None or label1.crs is None or label3.crs is None:
            raise SystemExit("至少一个文件缺少 CRS。")

        if image1.crs != label1.crs:
            raise SystemExit("Lushan_1 影像和标签声明的 CRS 不一致。")

        if label1.count != 1 or label3.count != 1:
            raise SystemExit("标签不是单波段，请先检查。")

        reference = stack.enter_context(
            WarpedVRT(
                label1,
                crs=image1.crs,
                transform=image1.transform,
                width=image1.width,
                height=image1.height,
                resampling=Resampling.nearest,
                add_alpha=True,
            )
        )

        shifted = {}

        for name, (dc, dr) in SHIFTS.items():
            shifted_transform = (
                image1.transform * Affine.translation(dc, dr)
            )

            shifted[name] = stack.enter_context(
                WarpedVRT(
                    label3,
                    crs="EPSG:4543",
                    transform=shifted_transform,
                    width=image1.width,
                    height=image1.height,
                    resampling=Resampling.nearest,
                    add_alpha=True,
                )
            )

        if reference.count != 2 or any(
            vrt.count != 2 for vrt in shifted.values()
        ):
            raise SystemExit("虚拟 alpha 波段结构异常。")

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

                first = reference.read(1, window=window)
                common = reference.read(2, window=window) > 0
                arrays = {}

                for name, vrt in shifted.items():
                    arrays[name] = vrt.read(1, window=window)
                    common &= vrt.read(2, window=window) > 0

                common_pixels += int(common.sum())
                first_positive = (first == 1) & common
                label1_positive += int(first_positive.sum())

                for name, array in arrays.items():
                    third_positive = (array == 1) & common

                    totals[name]["label3_positive"] += int(
                        third_positive.sum()
                    )
                    totals[name]["intersection"] += int(
                        (first_positive & third_positive).sum()
                    )
                    totals[name]["union"] += int(
                        (first_positive | third_positive).sum()
                    )

                done += 1
                if done % 25 == 0 or done == total_blocks:
                    print(f"已处理 {done}/{total_blocks} 个窗口", flush=True)

    if common_pixels == 0:
        raise SystemExit("没有五种情况共同覆盖的像素，未保存结果。")

    for name, values in totals.items():
        union = values["union"]
        values["foreground_iou"] = (
            values["intersection"] / union if union else None
        )
        values["shift_col_row_pixels"] = list(SHIFTS[name])

    report = {
        "status": "exploratory_not_crs_confirmation",
        "hypothesis": "Lushan_1 numeric grid interpreted as EPSG:4543",
        "common_footprint_pixels": common_pixels,
        "label1_positive_on_common_footprint": label1_positive,
        "controls": totals,
        "limitations": [
            "All controls use the same geometric/metadata footprint.",
            "Footprint is not a validated imagery-validity mask.",
            "Shift controls are not independent ground truth.",
            "Labels may share annotation sources.",
        ],
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n共同范围像素：{common_pixels:,}")
    print(f"共同范围内 Lushan_1 滑坡像素：{label1_positive:,}")

    for name, values in totals.items():
        iou = values["foreground_iou"]
        iou_text = f"{iou:.4f}" if iou is not None else "无法计算"

        print(
            f"  {name}：交集 {values['intersection']:,}，"
            f"并集 {values['union']:,}，IoU {iou_text}"
        )

    print(f"\n结果已保存：{OUTPUT}")
    print("注意：本结果不能单独证明 CRS 正确。")


if __name__ == "__main__":
    main()
