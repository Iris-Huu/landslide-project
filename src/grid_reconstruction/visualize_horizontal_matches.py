"""
可视化无人机切片的水平候选匹配。

每组展示：
1. 左侧影像；
2. 右侧影像；
3. 根据 50% 重叠关系拼接后的影像。

黄色线表示重叠区域的左右边界。
"""

from pathlib import Path
import csv
import warnings

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning


PROJECT_DIR = Path(__file__).resolve().parents[2]

IMAGE_DIR = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "CAS"
    / "Moxitaidi_UAV_0.6m"
    / "img"
)

MATCH_PATH = (
    PROJECT_DIR
    / "data"
    / "interim"
    / "reconstructed_grids"
    / "uav_horizontal_candidates.csv"
)

OUTPUT_PATH = (
    PROJECT_DIR
    / "reports"
    / "figures"
    / "uav_horizontal_matches_preview.png"
)

SAMPLE_COUNT = 3

warnings.filterwarnings(
    "ignore",
    category=NotGeoreferencedWarning,
)


def read_image(filename):
    """读取 RGB 影像并调整为绘图使用的形状。"""
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

    with MATCH_PATH.open(
        encoding="utf-8",
        newline="",
    ) as file:
        matches = list(csv.DictReader(file))

    selected = matches[:SAMPLE_COUNT]

    if not selected:
        raise SystemExit("候选表中没有匹配影像对，无法生成预览。")

    fig, axes = plt.subplots(
        len(selected),
        3,
        figsize=(16, 5 * len(selected)),
        squeeze=False,
    )

    for row, match in enumerate(selected):
        left_name = match["left_image"]
        right_name = match["right_image"]

        left_image = read_image(left_name)
        right_image = read_image(right_name)

        # 再次检查重叠区域是否逐像素相同
        if not np.array_equal(
            left_image[:, 256:],
            right_image[:, :256],
        ):
            raise ValueError(
                f"重叠区域不一致：{left_name} / {right_name}"
            )

        # 保留左图完整内容，再接上右图不重叠的右半部分
        stitched = np.concatenate(
            [left_image, right_image[:, 256:]],
            axis=1,
        )

        axes[row, 0].imshow(left_image)
        axes[row, 0].set_title(f"Left: {left_name}")

        axes[row, 1].imshow(right_image)
        axes[row, 1].set_title(f"Right: {right_name}")

        axes[row, 2].imshow(stitched)
        axes[row, 2].axvline(256, color="yellow", linewidth=1)
        axes[row, 2].axvline(512, color="yellow", linewidth=1)
        axes[row, 2].set_title("Stitched; overlap between lines")

        for ax in axes[row]:
            ax.axis("off")

        print(f"已核对并拼接：{left_name} -> {right_name}")

    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"\n预览图已保存：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
    