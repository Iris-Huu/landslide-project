"""
可视化滑坡标注与候选无数据区的重叠。

颜色说明：
- 红色：滑坡标注
- 蓝色：候选无数据区
- 黄色：两者重叠

注意：候选无数据区是待验证的规则结果，不是最终标注。
"""

from pathlib import Path
import warnings

import matplotlib

# AutoDL 服务器通常没有图形界面，使用 Agg 后端保存图片
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from scipy import ndimage


# CAS 切片可能不带地理坐标；本次只检查像素，因此隐藏该提示
warnings.filterwarnings(
    "ignore",
    category=NotGeoreferencedWarning,
)

PROJECT_DIR = Path("/root/autodl-tmp/landslide_project")

DATASET_DIR = (
    PROJECT_DIR
    / "data"
    / "raw"
    / "CAS"
    / "Moxitaidi_UAV_0.6m"
)

OUTPUT_DIR = PROJECT_DIR / "reports" / "figures"
OUTPUT_PATH = OUTPUT_DIR / "UAV_nodata_landslide_overlap.png"

# 根据前面的统计，选择重叠像素最多的 5 张无人机影像
FILENAMES = [
    "luding_UAV0641.tif",
    "luding_UAV0626.tif",
    "luding_UAV0642.tif",
    "luding_UAV0627.tif",
    "luding_UAV0640.tif",
]


def read_image_and_mask(filename):
    """读取一张 RGB 影像及其对应的滑坡掩码。"""

    image_path = DATASET_DIR / "img" / filename
    mask_path = DATASET_DIR / "mask" / filename

    with rasterio.open(image_path) as src:
        # rasterio 读取后为：通道 × 高 × 宽
        # 绘图需要：高 × 宽 × 通道
        image = np.moveaxis(src.read(), 0, -1)

    with rasterio.open(mask_path) as src:
        # True 表示掩码中标注为滑坡的像素
        landslide = src.read(1) == 1

    return image, landslide


def find_nodata_candidate(image):
    """找出与影像边缘连通的纯黑或纯白像素。"""

    pure_black = np.all(image == 0, axis=2)
    pure_white = np.all(image == 255, axis=2)

    possible = pure_black | pure_white

    # 将四条边上的候选像素设为搜索起点
    seeds = np.zeros(possible.shape, dtype=bool)

    seeds[0, :] = possible[0, :]
    seeds[-1, :] = possible[-1, :]
    seeds[:, 0] = possible[:, 0]
    seeds[:, -1] = possible[:, -1]

    # 从边缘出发，寻找上下左右连通的候选像素
    candidate = ndimage.binary_propagation(
        seeds,
        mask=possible,
    )

    return candidate


def make_overlay(landslide, candidate):
    """为滑坡、候选无数据区及重叠区域设置颜色。"""

    overlap = landslide & candidate

    # 最后一维的四个数依次为：红、绿、蓝、透明度
    overlay = np.zeros((*candidate.shape, 4), dtype=float)

    # 红色：滑坡标注，但不在候选无数据区内
    overlay[landslide & ~candidate] = [1, 0, 0, 0.65]

    # 蓝色：候选无数据区，但没有滑坡标注
    overlay[candidate & ~landslide] = [0, 0.3, 1, 0.65]

    # 黄色：滑坡标注与候选无数据区重叠
    overlay[overlap] = [1, 1, 0, 0.85]

    return overlay, overlap


def main():
    """读取样本，生成并保存对比图。"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        len(FILENAMES),
        2,
        figsize=(10, 20),
    )

    for row, filename in enumerate(FILENAMES):
        image, landslide = read_image_and_mask(filename)

        candidate = find_nodata_candidate(image)

        overlay, overlap = make_overlay(
            landslide,
            candidate,
        )

        # 左图：原始影像
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(filename)
        axes[row, 0].axis("off")

        # 右图：影像与标注叠加
        axes[row, 1].imshow(image)
        axes[row, 1].imshow(overlay)
        axes[row, 1].set_title(
            f"Overlap: {overlap.sum():,} pixels"
        )
        axes[row, 1].axis("off")

        print(
            f"{filename}: "
            f"重叠 {overlap.sum():,} 像素"
        )

    fig.tight_layout()
    fig.savefig(
        OUTPUT_PATH,
        dpi=130,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"\n图片已保存：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()