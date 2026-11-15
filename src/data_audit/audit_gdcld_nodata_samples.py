"""
抽样审计 GDCLD Lushan_1/2 的纯黑与窗口边缘连通候选区域。

注意：
- 每景在影像范围内均匀选取 7×7 个 512×512 窗口；
- 窗口边缘连通不等于整景影像边缘连通；
- 候选区域不是最终无数据掩码；
- 不修改原始数据。
"""

import csv
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from scipy import ndimage


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "data/raw/GDCLD/Lushan_extracted"

OUTPUT = (
    PROJECT / "reports/tables/"
    "gdcld_lushan_nodata_samples_v1.csv"
)

SCENES = ["Lushan_1.tif", "Lushan_2.tif"]
SIZE = 512
GRID = 7


def edge_connected_black(rgb):
    black = np.all(rgb == 0, axis=0)

    seeds = np.zeros(black.shape, dtype=bool)
    seeds[0, :] = black[0, :]
    seeds[-1, :] = black[-1, :]
    seeds[:, 0] = black[:, 0]
    seeds[:, -1] = black[:, -1]

    candidate = ndimage.binary_propagation(seeds, mask=black)
    return black, candidate


def sample_scene(name):
    image_path = ROOT / "test_data/Lushan" / name
    label_path = ROOT / "test_label/Lushan" / name

    results = []

    with rasterio.open(image_path) as image, rasterio.open(label_path) as label:
        if image.crs is None or label.crs is None:
            raise ValueError(f"{name} 缺少坐标系")

        if image.crs != label.crs:
            raise ValueError(f"{name} 影像与标签坐标系不同")

        if image.width < SIZE or image.height < SIZE or image.count < 3:
            raise ValueError(f"{name} 尺寸或波段数不足")

        rows = np.unique(
            np.linspace(0, image.height - SIZE, GRID, dtype=int)
        )
        cols = np.unique(
            np.linspace(0, image.width - SIZE, GRID, dtype=int)
        )

        with WarpedVRT(
            label,
            crs=image.crs,
            transform=image.transform,
            width=image.width,
            height=image.height,
            resampling=Resampling.nearest,
        ) as aligned:

            for row in rows:
                for col in cols:
                    window = Window(int(col), int(row), SIZE, SIZE)

                    rgb = image.read([1, 2, 3], window=window)
                    image_masks = image.read_masks(
                        [1, 2, 3], window=window
                    )
                    label_array = aligned.read(1, window=window)
                    label_mask = aligned.read_masks(1, window=window) > 0

                    black, candidate = edge_connected_black(rgb)
                    image_valid = np.all(image_masks > 0, axis=0)
                    landslide = (label_array == 1) & label_mask
                    overlap = candidate & landslide

                    results.append({
                        "scene": name,
                        "row": int(row),
                        "col": int(col),
                        "pixels": int(black.size),
                        "pure_black_pixels": int(black.sum()),
                        "candidate_pixels": int(candidate.sum()),
                        "candidate_marked_valid_by_image_mask": int(
                            (candidate & image_valid).sum()
                        ),
                        "landslide_pixels": int(landslide.sum()),
                        "candidate_landslide_overlap": int(overlap.sum()),
                    })

    return results


def main():
    if OUTPUT.exists():
        raise SystemExit(f"输出文件已存在，未覆盖：{OUTPUT}")

    results = []

    for name in SCENES:
        print(f"开始抽样：{name}", flush=True)
        rows = sample_scene(name)
        results.extend(rows)

        pixels = sum(row["pixels"] for row in rows)
        black = sum(row["pure_black_pixels"] for row in rows)
        candidate = sum(row["candidate_pixels"] for row in rows)
        landslide = sum(row["landslide_pixels"] for row in rows)
        overlap = sum(row["candidate_landslide_overlap"] for row in rows)

        print(f"  窗口数：{len(rows)}")
        print(f"  抽样窗口总像素：{pixels:,}")
        print(f"  纯黑像素：{black:,}（{black / pixels * 100:.2f}%）")
        print(
            f"  窗口边缘连通纯黑候选："
            f"{candidate:,}（{candidate / pixels * 100:.2f}%）"
        )
        print(f"  抽样窗口滑坡标注像素：{landslide:,}")
        print(f"  候选与滑坡标注重叠：{overlap:,}")

        if landslide:
            print(f"  重叠占抽样滑坡像素：{overlap / landslide * 100:.4f}%")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    print(f"\n结果已保存：{OUTPUT}")
    print("注意：这是抽样结果，候选区域不是最终无数据掩码。")


if __name__ == "__main__":
    main()
