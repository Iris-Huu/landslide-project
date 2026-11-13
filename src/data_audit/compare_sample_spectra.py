"""
对一组已做局部配准的卫星与无人机影像进行探索性频谱比较。

注意：
- 使用一个固定的 256×256 像素局部窗口；
- 卫星影像经过插值，会影响频谱；
- 输出为相对频谱，不是以米为单位的有效分辨率；
- 不修改原始影像。
"""

import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rasterio


PROJECT = Path(__file__).resolve().parents[2]

SAT_PATH = (
    PROJECT / "data/raw/CAS/Moxitaidi_SAT/img/"
    "Moxitaidi_Satellite001.tif"
)
UAV_PATH = (
    PROJECT / "data/raw/CAS/Moxitaidi_UAV_0.6m/img/"
    "luding_UAV0011.tif"
)
TRANSFORM_PATH = (
    PROJECT / "data/interim/registration/"
    "sat001_uav0011_min_inliers_check_result.json"
)

FIGURE_PATH = (
    PROJECT / "reports/figures/"
    "sat001_uav0011_spectral_comparison.png"
)
RESULT_PATH = (
    PROJECT / "data/interim/registration/"
    "sat001_uav0011_spectral_comparison.json"
)


def read_rgb(path):
    with rasterio.open(path) as src:
        image = src.read()

    if image.shape != (3, 512, 512):
        raise ValueError(f"影像形状异常：{path}，{image.shape}")

    return np.moveaxis(image, 0, -1)


def radial_power(gray):
    """计算加窗后的二维功率谱及径向平均。"""
    image = gray.astype(np.float64)
    image -= image.mean()

    window = np.outer(
        np.hanning(image.shape[0]),
        np.hanning(image.shape[1]),
    )

    spectrum = np.fft.fftshift(np.fft.fft2(image * window))
    power = np.abs(spectrum) ** 2

    fy = np.fft.fftshift(np.fft.fftfreq(image.shape[0]))
    fx = np.fft.fftshift(np.fft.fftfreq(image.shape[1]))
    radius = np.hypot(fy[:, None], fx[None, :])

    edges = np.linspace(0.0, 0.5, 65)
    centers = (edges[:-1] + edges[1:]) / 2
    radial = np.full(len(centers), np.nan)

    for index in range(len(centers)):
        selected = (radius >= edges[index]) & (radius < edges[index + 1])
        if selected.any():
            radial[index] = power[selected].mean()

    return centers, radial


def main():
    for path in [FIGURE_PATH, RESULT_PATH]:
        if path.exists():
            raise SystemExit(f"输出文件已存在，未覆盖：{path}")

    with TRANSFORM_PATH.open(encoding="utf-8") as f:
        registration = json.load(f)

    matrix = np.asarray(
        registration["transform_satellite_to_uav"],
        dtype=np.float64,
    )

    if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
        raise ValueError("配准矩阵不存在或格式异常")

    satellite = read_rgb(SAT_PATH)
    uav = read_rgb(UAV_PATH)

    # 把卫星图变换到 UAV 像素坐标系。
    warped = cv2.warpAffine(
        satellite,
        matrix,
        (512, 512),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    # 几何覆盖：只表示坐标来自卫星图范围，不表示像素一定有效。
    coverage = cv2.warpAffine(
        np.ones((512, 512), dtype=np.uint8),
        matrix,
        (512, 512),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # UAV 坐标中的中心窗口：行、列均为 128:384。
    window = (slice(128, 384), slice(128, 384))
    sat_crop = warped[window]
    uav_crop = uav[window]
    geometric_coverage = float(coverage[window].mean())

    if geometric_coverage < 1.0:
        raise SystemExit(
            f"窗口几何覆盖不足 100%：{geometric_coverage:.2%}；"
            "请检查窗口位置。"
        )

    sat_gray = cv2.cvtColor(sat_crop, cv2.COLOR_RGB2GRAY)
    uav_gray = cv2.cvtColor(uav_crop, cv2.COLOR_RGB2GRAY)

    frequency, sat_power = radial_power(sat_gray)
    _, uav_power = radial_power(uav_gray)

    # 按各自的低频平均功率归一化，便于比较曲线形状。
    reference = (frequency >= 0.02) & (frequency < 0.08)

    sat_reference = float(np.nanmean(sat_power[reference]))
    uav_reference = float(np.nanmean(uav_power[reference]))

    if sat_reference <= 0 or uav_reference <= 0:
        raise ValueError("低频参考功率异常，无法归一化。")

    sat_normalized = sat_power / sat_reference
    uav_normalized = uav_power / uav_reference

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    axes[0].imshow(sat_crop)
    axes[0].set_title("Satellite, warped to UAV grid")
    axes[0].axis("off")

    axes[1].imshow(uav_crop)
    axes[1].set_title("UAV: same pixel window")
    axes[1].axis("off")

    axes[2].semilogy(
        frequency, sat_normalized, label="Satellite warped"
    )
    axes[2].semilogy(
        frequency, uav_normalized, label="UAV"
    )
    axes[2].set_xlabel("Frequency (cycles / UAV pixel)")
    axes[2].set_ylabel("Radial power / low-frequency mean")
    axes[2].set_title("Exploratory relative spectra")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    fig.tight_layout()

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(FIGURE_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)

    result = {
        "satellite": SAT_PATH.name,
        "uav": UAV_PATH.name,
        "registration_record": str(TRANSFORM_PATH.relative_to(PROJECT)),
        "window_uav_pixels": {
            "row_start": 128,
            "row_end_exclusive": 384,
            "col_start": 128,
            "col_end_exclusive": 384,
        },
        "geometric_coverage_fraction": geometric_coverage,
        "normalization_frequency_cycles_per_uav_pixel": [0.02, 0.08],
        "frequency_cycles_per_uav_pixel": frequency.tolist(),
        "satellite_normalized_power": sat_normalized.tolist(),
        "uav_normalized_power": uav_normalized.tolist(),
        "limitation": (
            "Exploratory comparison for one crop; interpolation and "
            "sensor/content differences affect spectra. Not an absolute "
            "effective-resolution measurement."
        ),
    }

    with RESULT_PATH.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)

    print(f"窗口几何覆盖：{geometric_coverage:.2%}")
    print(f"对照图：{FIGURE_PATH}")
    print(f"频谱记录：{RESULT_PATH}")
    print("注意：频谱差异不能直接换算为实际有效分辨率。")


if __name__ == "__main__":
    main()
