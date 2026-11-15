"""
统计 GDCLD Lushan_1/2 中与整景边界连通的纯黑候选区域。

注意：
- 候选区域不是已确认的无数据掩码；
- 不修改原始数据；
- 标签按最近邻虚拟对齐到影像网格；
- 四邻域连通，不包含仅通过对角点相连的区域。
"""

import csv
import gc
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from scipy import ndimage


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "data/raw/GDCLD/Lushan_extracted"
OUTPUT = PROJECT / "reports/tables/gdcld_lushan_scene_edge_black_v1.csv"

SCENES = ["Lushan_1.tif", "Lushan_2.tif"]
BLOCK = 2048
MIN_AVAILABLE = 3 * 1024 ** 3


def available_memory():
    values = {}

    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                values["system"] = int(line.split()[1]) * 1024

    # 兼容常见的 cgroup v2 内存限制。
    maximum = Path("/sys/fs/cgroup/memory.max")
    current = Path("/sys/fs/cgroup/memory.current")

    if maximum.exists() and current.exists():
        limit_text = maximum.read_text().strip()

        if limit_text != "max":
            remaining = int(limit_text) - int(current.read_text().strip())
            values["cgroup"] = max(remaining, 0)

    return min(values.values()) if values else None


def windows(width, height):
    for row in range(0, height, BLOCK):
        for col in range(0, width, BLOCK):
            yield Window(
                col,
                row,
                min(BLOCK, width - col),
                min(BLOCK, height - row),
            )


def inspect_scene(name):
    image_path = ROOT / "test_data/Lushan" / name
    label_path = ROOT / "test_label/Lushan" / name

    with rasterio.open(image_path) as image, rasterio.open(label_path) as label:
        if image.crs is None or label.crs is None:
            raise ValueError(f"{name} 缺少 CRS")

        if image.crs != label.crs:
            raise ValueError(f"{name} 影像和标签 CRS 不一致")

        if image.count < 3:
            raise ValueError(f"{name} 影像不足 3 个波段")

        black = np.zeros((image.height, image.width), dtype=bool)

        print(f"  读取纯黑像素：{name}", flush=True)

        for window in windows(image.width, image.height):
            rgb = image.read([1, 2, 3], window=window)

            row = int(window.row_off)
            col = int(window.col_off)
            height = int(window.height)
            width = int(window.width)

            black[row:row + height, col:col + width] = np.all(
                rgb == 0, axis=0
            )

        seeds = np.zeros_like(black)
        seeds[0, :] = black[0, :]
        seeds[-1, :] = black[-1, :]
        seeds[:, 0] = black[:, 0]
        seeds[:, -1] = black[:, -1]

        print("  计算整景边界连通区域……", flush=True)

        candidate = ndimage.binary_propagation(
            seeds,
            structure=ndimage.generate_binary_structure(2, 1),
            mask=black,
        )

        del seeds

        total_landslide = 0
        candidate_landslide = 0

        with WarpedVRT(
            label,
            crs=image.crs,
            transform=image.transform,
            width=image.width,
            height=image.height,
            resampling=Resampling.nearest,
        ) as aligned:

            for window in windows(image.width, image.height):
                row = int(window.row_off)
                col = int(window.col_off)
                height = int(window.height)
                width = int(window.width)

                label_array = aligned.read(1, window=window)
                label_valid = aligned.read_masks(1, window=window) > 0
                landslide = (label_array == 1) & label_valid

                candidate_window = candidate[
                    row:row + height,
                    col:col + width,
                ]

                total_landslide += int(landslide.sum())
                candidate_landslide += int(
                    (candidate_window & landslide).sum()
                )

        result = {
            "scene": name,
            "image_pixels": int(black.size),
            "pure_black_pixels": int(black.sum()),
            "scene_edge_connected_black_pixels": int(candidate.sum()),
            "aligned_landslide_pixels": total_landslide,
            "candidate_landslide_overlap_pixels": candidate_landslide,
        }

        del black, candidate
        gc.collect()

    return result


def main():
    if OUTPUT.exists():
        raise SystemExit(f"输出文件已存在，未覆盖：{OUTPUT}")

    available = available_memory()

    if available is None:
        raise SystemExit("无法确认可用内存，为避免内存不足已停止。")

    print(f"检测到可用内存约：{available / 1024 ** 3:.2f} GiB")

    if available < MIN_AVAILABLE:
        raise SystemExit("可用内存不足 3 GiB，已停止。")

    results = []

    for name in SCENES:
        print(f"\n开始：{name}", flush=True)
        result = inspect_scene(name)
        results.append(result)

        total = result["image_pixels"]
        black = result["pure_black_pixels"]
        candidate = result["scene_edge_connected_black_pixels"]
        landslide = result["aligned_landslide_pixels"]
        overlap = result["candidate_landslide_overlap_pixels"]

        print(f"  整景像素：{total:,}")
        print(f"  纯黑像素：{black:,}（{black / total * 100:.2f}%）")
        print(
            f"  连通整景边界的纯黑候选："
            f"{candidate:,}（{candidate / total * 100:.2f}%）"
        )
        print(f"  与滑坡标注重叠：{overlap:,}")

        if landslide:
            print(f"  重叠占滑坡标注：{overlap / landslide * 100:.4f}%")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    print(f"\n结果已保存：{OUTPUT}")
    print("注意：候选区域不是最终无数据掩码。")


if __name__ == "__main__":
    main()
