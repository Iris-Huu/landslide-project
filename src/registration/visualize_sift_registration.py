"""
可视化一对 Moxitaidi 样本的 SIFT 相似变换。

输出：
1. 无人机原图、变换后的卫星图、叠加图和棋盘格图；
2. 本次变换矩阵和基础统计的 JSON 文件。

注意：
- 几何覆盖范围不等于有效影像范围；
- 本程序不判断纯黑或纯白像素是否为无数据；
- 本次结果仍需人工和独立地物验证。
"""

from pathlib import Path
import json

import cv2
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# 复用通用匹配程序的数据目录、参数与读取函数
from match_sift_pair import (
    SAT_DIR,
    UAV_DIR,
    RATIO_THRESHOLD,
    RANSAC_THRESHOLD,
    read_rgb as read_rgb_from_dir,
)

SATELLITE_PATH = SAT_DIR / "Moxitaidi_Satellite001.tif"
UAV_PATH = UAV_DIR / "luding_UAV0001.tif"


def read_rgb(path):
    return read_rgb_from_dir(path.parent, path.name)


PROJECT_DIR = Path(__file__).resolve().parents[2]

FIGURE_PATH = (
    PROJECT_DIR
    / "reports/figures"
    / "moxitaidi_sift_registration_overlay.png"
)

RESULT_PATH = (
    PROJECT_DIR
    / "data/interim/registration"
    / "moxitaidi_sift_sample_transform.json"
)

CHECKER_SIZE = 64


def estimate_transform(satellite, uav):
    """重新提取特征并估计卫星到无人机的相似变换。"""
    sat_gray = cv2.cvtColor(satellite, cv2.COLOR_RGB2GRAY)
    uav_gray = cv2.cvtColor(uav, cv2.COLOR_RGB2GRAY)

    sift = cv2.SIFT_create()
    sat_points, sat_descriptors = sift.detectAndCompute(sat_gray, None)
    uav_points, uav_descriptors = sift.detectAndCompute(uav_gray, None)

    if sat_descriptors is None or uav_descriptors is None:
        raise RuntimeError("至少一张影像没有生成 SIFT 描述。")

    if len(uav_descriptors) < 2:
        raise RuntimeError("无人机特征描述不足两个，无法进行比例筛选。")

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    nearest_pairs = matcher.knnMatch(
        sat_descriptors,
        uav_descriptors,
        k=2,
    )

    candidates = []

    for neighbors in nearest_pairs:
        if len(neighbors) < 2:
            continue

        best, second = neighbors
        if best.distance < RATIO_THRESHOLD * second.distance:
            candidates.append(best)

    if len(candidates) < 4:
        raise RuntimeError(
            f"候选匹配只有 {len(candidates)} 对，"
            "不足以进行本次变换估计。"
        )

    satellite_xy = np.float32([
        sat_points[match.queryIdx].pt
        for match in candidates
    ]).reshape(-1, 1, 2)

    uav_xy = np.float32([
        uav_points[match.trainIdx].pt
        for match in candidates
    ]).reshape(-1, 1, 2)

    cv2.setRNGSeed(42)

    matrix, inlier_mask = cv2.estimateAffinePartial2D(
        satellite_xy,
        uav_xy,
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_THRESHOLD,
        maxIters=5000,
        confidence=0.99,
    )

    if matrix is None or inlier_mask is None:
        raise RuntimeError("未能估计出相似变换。")

    inlier_flags = inlier_mask.ravel().astype(bool)
    inlier_count = int(inlier_flags.sum())

    if inlier_count == 0:
        raise RuntimeError("变换没有内点，无法计算误差。")

    source = satellite_xy.reshape(-1, 2)[inlier_flags]
    target = uav_xy.reshape(-1, 2)[inlier_flags]

    predicted = source @ matrix[:, :2].T + matrix[:, 2]
    residuals = np.linalg.norm(predicted - target, axis=1)

    return matrix, {
        "satellite_keypoints": len(sat_points),
        "uav_keypoints": len(uav_points),
        "candidate_matches": len(candidates),
        "ransac_inliers": inlier_count,
        "median_inlier_error_uav_pixels": float(
            np.median(residuals)
        ),
    }


def main():
    for path in [FIGURE_PATH, RESULT_PATH]:
        if path.exists():
            raise SystemExit(
                f"输出文件已存在，为避免覆盖而停止：{path}"
            )

    satellite = read_rgb(SATELLITE_PATH)
    uav = read_rgb(UAV_PATH)

    matrix, stats = estimate_transform(satellite, uav)

    height, width = uav.shape[:2]

    # 将卫星影像变换到无人机像素坐标系
    warped_satellite = cv2.warpAffine(
        satellite,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    # 只标记变换后是否落在卫星图的几何范围内。
    # 这不是经过验证的无数据区掩码。
    source_footprint = np.ones(
        satellite.shape[:2],
        dtype=np.uint8,
    )

    warped_footprint = cv2.warpAffine(
        source_footprint,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)

    # 为了展示几何范围外的位置，用灰色作为画布背景
    warped_display = warped_satellite.copy()
    warped_display[~warped_footprint] = [128, 128, 128]

    # 几何覆盖内做半透明叠加；覆盖外保留无人机原图
    blended = uav.copy()
    mixed = cv2.addWeighted(
        uav,
        0.5,
        warped_satellite,
        0.5,
        0,
    )
    blended[warped_footprint] = mixed[warped_footprint]

    # 棋盘格：交替显示无人机与变换后的卫星影像
    rows, cols = np.indices((height, width))
    use_satellite = (
        (rows // CHECKER_SIZE + cols // CHECKER_SIZE) % 2 == 0
    ) & warped_footprint

    checkerboard = uav.copy()
    checkerboard[use_satellite] = warped_satellite[use_satellite]

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    axes[0, 0].imshow(uav)
    axes[0, 0].set_title("UAV image")

    axes[0, 1].imshow(warped_display)
    axes[0, 1].set_title("Satellite warped to UAV coordinates")

    axes[1, 0].imshow(blended)
    axes[1, 0].set_title("50% blend within geometric coverage")

    axes[1, 1].imshow(checkerboard)
    axes[1, 1].set_title("Checkerboard comparison")

    for ax in axes.ravel():
        ax.axis("off")

    fig.suptitle(
        "Exploratory SIFT registration — sample pair"
    )
    fig.tight_layout()

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(FIGURE_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    result = {
        "source_image": SATELLITE_PATH.name,
        "target_image": UAV_PATH.name,
        "transform_direction": "satellite_pixels_to_uav_pixels",
        "ratio_threshold": RATIO_THRESHOLD,
        "ransac_threshold_uav_pixels": RANSAC_THRESHOLD,
        "random_seed": 42,
        "matrix_2x3": matrix.tolist(),
        "geometric_coverage_percent": float(
            warped_footprint.mean() * 100
        ),
        "statistics": stats,
        "note": (
            "Exploratory result. Geometric coverage is not a "
            "validated no-data mask. Inlier error is not an "
            "independent accuracy assessment."
        ),
    }

    with RESULT_PATH.open("x", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    print("\n重新估计的相似变换矩阵：")
    print(matrix)
    print(f"候选匹配：{stats['candidate_matches']} 对")
    print(f"RANSAC 内点：{stats['ransac_inliers']} 对")
    print(
        "内点误差中位数："
        f"{stats['median_inlier_error_uav_pixels']:.2f} 像素"
    )
    print(
        "卫星图在无人机画布上的几何覆盖："
        f"{warped_footprint.mean() * 100:.2f}%"
    )
    print(f"\n对照图：{FIGURE_PATH}")
    print(f"变换记录：{RESULT_PATH}")


if __name__ == "__main__":
    main()