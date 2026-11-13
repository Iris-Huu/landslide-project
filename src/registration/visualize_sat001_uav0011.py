"""
可视化 Satellite001 → UAV0011 的局部配准结果。

读取已保存的相似变换矩阵，生成叠加图与棋盘格图。
本图用于目视检查，不是独立配准精度评估。
"""

from pathlib import Path
import json
import warnings

import cv2
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning


PROJECT_DIR = Path(__file__).resolve().parents[2]

SATELLITE_PATH = (
    PROJECT_DIR / "data/raw/CAS/Moxitaidi_SAT/img"
    / "Moxitaidi_Satellite001.tif"
)

UAV_PATH = (
    PROJECT_DIR / "data/raw/CAS/Moxitaidi_UAV_0.6m/img"
    / "luding_UAV0011.tif"
)

RESULT_PATH = (
    PROJECT_DIR / "data/interim/registration"
    / "sat001_uav0011_result.json"
)

OUTPUT_PATH = (
    PROJECT_DIR / "reports/figures"
    / "sat001_uav0011_registration_overlay.png"
)

CHECKER_SIZE = 64

warnings.filterwarnings(
    "ignore",
    category=NotGeoreferencedWarning,
)


def read_rgb(path):
    """读取 RGB 影像并检查尺寸、类型。"""
    with rasterio.open(path) as src:
        image = src.read()

    if image.shape != (3, 512, 512):
        raise ValueError(f"{path.name} 的形状异常：{image.shape}")

    if image.dtype != np.uint8:
        raise ValueError(f"{path.name} 的类型不是 uint8")

    return np.moveaxis(image, 0, -1).copy()


def main():
    if OUTPUT_PATH.exists():
        raise SystemExit(
            f"输出图片已存在，为避免覆盖而停止：{OUTPUT_PATH}"
        )

    with RESULT_PATH.open(encoding="utf-8") as file:
        result = json.load(file)

    if (
        result["satellite_image"] != SATELLITE_PATH.name
        or result["uav_image"] != UAV_PATH.name
    ):
        raise ValueError("结果记录中的影像名称与当前样本不一致。")

    if result["transform_satellite_to_uav"] is None:
        raise ValueError("结果记录中没有可用的相似变换矩阵。")

    matrix = np.asarray(
        result["transform_satellite_to_uav"],
        dtype=float,
    )

    if matrix.shape != (2, 3):
        raise ValueError(f"变换矩阵形状异常：{matrix.shape}")

    satellite = read_rgb(SATELLITE_PATH)
    uav = read_rgb(UAV_PATH)

    height, width = uav.shape[:2]

    warped_satellite = cv2.warpAffine(
        satellite,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    # 几何覆盖范围不是有效影像掩码
    footprint = np.ones(
        satellite.shape[:2],
        dtype=np.uint8,
    )

    warped_footprint = cv2.warpAffine(
        footprint,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)

    # 几何范围外显示灰色
    satellite_display = warped_satellite.copy()
    satellite_display[~warped_footprint] = [128, 128, 128]

    # 范围内叠加；范围外仍显示无人机影像
    blended = uav.copy()
    mixed = cv2.addWeighted(
        uav, 0.5, warped_satellite, 0.5, 0
    )
    blended[warped_footprint] = mixed[warped_footprint]

    # 棋盘格交替显示两幅影像
    rows, cols = np.indices((height, width))
    use_satellite = (
        (rows // CHECKER_SIZE + cols // CHECKER_SIZE) % 2 == 0
    ) & warped_footprint

    checkerboard = uav.copy()
    checkerboard[use_satellite] = warped_satellite[use_satellite]

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    axes[0, 0].imshow(uav)
    axes[0, 0].set_title("UAV0011")

    axes[0, 1].imshow(satellite_display)
    axes[0, 1].set_title("Satellite001 warped to UAV0011")

    axes[1, 0].imshow(blended)
    axes[1, 0].set_title("50% blend within geometric coverage")

    axes[1, 1].imshow(checkerboard)
    axes[1, 1].set_title("Checkerboard comparison")

    for ax in axes.ravel():
        ax.axis("off")

    fig.suptitle("Exploratory registration: Satellite001 → UAV0011")
    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("使用的变换矩阵：")
    print(matrix)
    print(
        "卫星图在无人机画布上的几何覆盖："
        f"{warped_footprint.mean() * 100:.2f}%"
    )
    print(f"对照图已保存：{OUTPUT_PATH}")
    print(
        "注意：几何覆盖不等于有效影像范围；"
        "本图也不是独立精度评估。"
    )


if __name__ == "__main__":
    main()