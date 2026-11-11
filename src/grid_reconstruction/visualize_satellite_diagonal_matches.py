"""
可视化两组右下方向的对角候选匹配。

拼接图中的灰色区域表示这两张切片没有覆盖的位置，
不是对原始影像无数据区的判断。
"""

from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning


PROJECT = Path(__file__).resolve().parents[2]

IMAGE_DIR = (
    PROJECT
    / "data"
    / "raw"
    / "CAS"
    / "Moxitaidi_SAT"
    / "img"
)

OUTPUT_PATH = (
    PROJECT
    / "reports"
    / "figures"
    / "satellite_cross_component_diagonal_preview.png"
)

PAIRS = [
    ("Moxitaidi_Satellite245.tif", "Moxitaidi_Satellite251.tif"),
    ("Moxitaidi_Satellite263.tif", "Moxitaidi_Satellite264.tif"),
    ("Moxitaidi_Satellite548.tif", "Moxitaidi_Satellite551.tif"),
    ("Moxitaidi_Satellite557.tif", "Moxitaidi_Satellite563.tif"),
    ("Moxitaidi_Satellite582.tif", "Moxitaidi_Satellite593.tif"),
]

warnings.filterwarnings(
    "ignore",
    category=NotGeoreferencedWarning,
)


def read_image(filename):
    with rasterio.open(IMAGE_DIR / filename) as src:
        image = src.read()

    if image.shape != (3, 512, 512):
        raise ValueError(
            f"{filename} 的形状异常：{image.shape}"
        )

    return np.moveaxis(image, 0, -1)


def main():
    if OUTPUT_PATH.exists():
        raise SystemExit(
            f"输出图片已存在，为避免覆盖而停止：{OUTPUT_PATH}"
        )

    fig, axes = plt.subplots(
        len(PAIRS),
        3,
        figsize=(16, 5 * len(PAIRS)),
        squeeze=False,
    )

    for row, (source_name, target_name) in enumerate(PAIRS):
        source = read_image(source_name)
        target = read_image(target_name)

        # 起点右下角应与目标左上角完全一致
        source_overlap = source[256:, 256:]
        target_overlap = target[:256, :256]

        if not np.array_equal(source_overlap, target_overlap):
            raise ValueError(
                f"重叠区域不一致：{source_name} / {target_name}"
            )

        # 构建 768×768 画布；灰色表示没有被这两张图覆盖
        canvas = np.full((768, 768, 3), 128, dtype=np.uint8)

        canvas[:512, :512] = source
        canvas[256:768, 256:768] = target

        axes[row, 0].imshow(source)
        axes[row, 0].set_title(f"Source: {source_name}")

        axes[row, 1].imshow(target)
        axes[row, 1].set_title(f"Target: {target_name}")

        axes[row, 2].imshow(canvas)
        axes[row, 2].set_title("Diagonal placement")

        # 黄框标出 256×256 的共同区域
        rectangle = plt.Rectangle(
            (256, 256),
            256,
            256,
            fill=False,
            edgecolor="yellow",
            linewidth=1.5,
        )
        axes[row, 2].add_patch(rectangle)

        for ax in axes[row]:
            ax.axis("off")

        print(f"已核对：{source_name} -> {target_name}")

    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n预览图已保存：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()