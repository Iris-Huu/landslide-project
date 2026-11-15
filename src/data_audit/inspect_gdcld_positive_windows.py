"""
查找 GDCLD Lushan_1/2 的含滑坡窗口并生成对齐预览。

标签通过 WarpedVRT 使用最近邻重采样到影像网格。
本程序不修改原始影像或标签；预览不能独立证明标注精度。
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "data/raw/GDCLD/Lushan_extracted"

FIGURE = (
    PROJECT / "reports/figures/"
    "gdcld_lushan_positive_alignment.png"
)
RESULT = (
    PROJECT / "reports/tables/"
    "gdcld_lushan_positive_alignment.json"
)

SCENES = ["Lushan_1.tif", "Lushan_2.tif"]
SIZE = 512


def inspect_scene(name):
    image_path = ROOT / "test_data/Lushan" / name
    label_path = ROOT / "test_label/Lushan" / name

    with rasterio.open(image_path) as image, rasterio.open(label_path) as label:
        if image.crs is None or label.crs is None:
            raise ValueError(f"{name} 缺少坐标系")

        if image.crs != label.crs:
            raise ValueError(f"{name} 影像和标签坐标系不同，需单独处理")

        if image.count < 3:
            raise ValueError(f"{name} 影像不足 3 个波段")

        if image.width < SIZE or image.height < SIZE:
            raise ValueError(f"{name} 影像尺寸不足 {SIZE}")

        # VRT 将标签按最近邻映射到影像的 CRS、变换和尺寸。
        with WarpedVRT(
            label,
            crs=image.crs,
            transform=image.transform,
            width=image.width,
            height=image.height,
            resampling=Resampling.nearest,
        ) as aligned:

            for label_row in range(0, label.height, SIZE):
                for label_col in range(0, label.width, SIZE):
                    width = min(SIZE, label.width - label_col)
                    height = min(SIZE, label.height - label_row)

                    label_window = Window(
                        label_col, label_row, width, height
                    )
                    sample = label.read(1, window=label_window)

                    if not np.any(sample == 1):
                        continue

                    # 将候选标签窗口中心转换到影像像素位置。
                    center_row = label_row + height // 2
                    center_col = label_col + width // 2
                    x, y = label.xy(center_row, center_col)
                    image_row, image_col = image.index(x, y)

                    top = max(
                        0,
                        min(int(image_row) - SIZE // 2, image.height - SIZE),
                    )
                    left = max(
                        0,
                        min(int(image_col) - SIZE // 2, image.width - SIZE),
                    )

                    image_window = Window(left, top, SIZE, SIZE)
                    aligned_label = aligned.read(1, window=image_window)
                    valid_mask = aligned.read_masks(1, window=image_window) > 0
                    positive = (aligned_label == 1) & valid_mask

                    if not positive.any():
                        continue

                    rgb = np.moveaxis(
                        image.read([1, 2, 3], window=image_window),
                        0,
                        -1,
                    )

                    metadata = {
                        "scene": name,
                        "image_window_row": top,
                        "image_window_col": left,
                        "window_size": SIZE,
                        "positive_pixels": int(positive.sum()),
                        "aligned_label_values": [
                            int(value) for value in np.unique(aligned_label)
                        ],
                        "vrt_mask_valid_percent": float(valid_mask.mean() * 100),
                    }

                    return rgb, positive, metadata

    raise RuntimeError(f"{name} 未找到可预览的含滑坡窗口")


def main():
    for path in [FIGURE, RESULT]:
        if path.exists():
            raise SystemExit(f"输出已存在，未覆盖：{path}")

    items = []

    for name in SCENES:
        print(f"正在查找 {name} 的含滑坡窗口……", flush=True)
        items.append(inspect_scene(name))

    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    RESULT.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(items), 2, figsize=(11, 11))

    for index, (rgb, positive, metadata) in enumerate(items):
        axes[index, 0].imshow(rgb)
        axes[index, 0].set_title(f"{metadata['scene']}: image")
        axes[index, 0].axis("off")

        overlay = np.zeros((*positive.shape, 4), dtype=np.float32)
        overlay[positive] = [1.0, 0.0, 0.0, 0.55]

        axes[index, 1].imshow(rgb)
        axes[index, 1].imshow(overlay)
        axes[index, 1].set_title(
            f"Aligned label: {metadata['positive_pixels']:,} pixels"
        )
        axes[index, 1].axis("off")

        print(f"\n{metadata['scene']}")
        print(
            f"  影像窗口左上角："
            f"行 {metadata['image_window_row']}，"
            f"列 {metadata['image_window_col']}"
        )
        print(f"  滑坡标注像素：{metadata['positive_pixels']:,}")
        print(f"  对齐后标签取值：{metadata['aligned_label_values']}")
        print(
            f"  VRT 有效性掩码覆盖："
            f"{metadata['vrt_mask_valid_percent']:.2f}%"
        )

    fig.suptitle("GDCLD Lushan: image and virtually aligned label")
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=150, bbox_inches="tight")
    plt.close(fig)

    report = {
        "scenes": [item[2] for item in items],
        "method": "WarpedVRT nearest-neighbor to image grid",
        "note": (
            "Visual inspection is not independent accuracy validation. "
            "VRT mask validity depends on source metadata."
        ),
    }

    with RESULT.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n预览图：{FIGURE}")
    print(f"记录：{RESULT}")
    print(
        "注意：VRT 掩码依赖源文件元数据，"
        "不能单独证明有效区或标注精度。"
    )


if __name__ == "__main__":
    main()
