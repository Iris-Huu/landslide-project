"""
对 Satellite001 → UAV0011 做空间留出检查。

流程：
1. 重新提取 SIFT 特征并进行比例筛选；
2. 限制每个无人机特征只对应一个卫星特征；
3. 按卫星特征点的 x 坐标分成左右两组；
4. 左边拟合、右边检查；然后交换。

注意：
- 留出点是自动匹配候选，可能包含错误匹配；
- 留出点不参与对应那一次的变换拟合；
- 结果不是人工控制点的独立配准精度；
- 不使用尚未验证的无数据区掩码。
"""

from pathlib import Path
import json

import cv2
import numpy as np

# 复用通用匹配脚本的数据目录与图像读取函数
from match_sift_pair import SAT_DIR, UAV_DIR, read_rgb


PROJECT_DIR = Path(__file__).resolve().parents[2]

OUTPUT_PATH = (
    PROJECT_DIR
    / "data/interim/registration"
    / "sat001_uav0011_spatial_holdout.json"
)

SATELLITE_NAME = "Moxitaidi_Satellite001.tif"
UAV_NAME = "luding_UAV0011.tif"

RATIO_THRESHOLD = 0.75
RANSAC_THRESHOLD = 3.0
RANDOM_SEED = 42
MIN_TRAIN_POINTS = 4


def predict(points, matrix):
    """使用 2×3 矩阵变换一组 (x, y) 坐标。"""
    return points @ matrix[:, :2].T + matrix[:, 2]


def evaluate_split(
    name,
    source_points,
    target_points,
    train_flags,
    test_flags,
):
    """使用一组点拟合，并在另一组候选点上检查。"""
    train_source = source_points[train_flags]
    train_target = target_points[train_flags]

    test_source = source_points[test_flags]
    test_target = target_points[test_flags]

    result = {
        "name": name,
        "training_candidates": len(train_source),
        "holdout_candidates": len(test_source),
        "status": "not_estimated",
        "training_ransac_inliers": None,
        "holdout_median_error_uav_pixels": None,
        "holdout_90th_percentile_error_uav_pixels": None,
        "holdout_within_3_pixels_percent": None,
        "transform_satellite_to_uav": None,
    }

    print(f"\n{name}")
    print(f"  拟合组候选点：{len(train_source)} 对")
    print(f"  留出组候选点：{len(test_source)} 对")

    if len(train_source) < MIN_TRAIN_POINTS or len(test_source) == 0:
        print("  点数不足，本方向未估计变换。")
        return result

    cv2.setRNGSeed(RANDOM_SEED)

    matrix, inlier_mask = cv2.estimateAffinePartial2D(
        train_source.reshape(-1, 1, 2),
        train_target.reshape(-1, 1, 2),
        method=cv2.RANSAC,
        ransacReprojThreshold=RANSAC_THRESHOLD,
        maxIters=5000,
        confidence=0.99,
    )

    if matrix is None or inlier_mask is None:
        print("  未能估计出变换。")
        return result

    predicted = predict(test_source, matrix)
    errors = np.linalg.norm(predicted - test_target, axis=1)

    training_inliers = int(inlier_mask.ravel().astype(bool).sum())
    median_error = float(np.median(errors))
    p90_error = float(np.percentile(errors, 90))
    within_3 = float(np.mean(errors <= 3.0) * 100)

    result.update({
        "status": "estimated_requires_review",
        "training_ransac_inliers": training_inliers,
        "holdout_median_error_uav_pixels": median_error,
        "holdout_90th_percentile_error_uav_pixels": p90_error,
        "holdout_within_3_pixels_percent": within_3,
        "transform_satellite_to_uav": matrix.tolist(),
    })

    print(f"  拟合组 RANSAC 内点：{training_inliers} 对")
    print(f"  留出组误差中位数：{median_error:.2f} 个无人机像素")
    print(f"  留出组误差 90% 分位数：{p90_error:.2f} 个无人机像素")
    print(f"  留出误差不超过 3 像素：{within_3:.2f}%")

    return result


def main():
    if OUTPUT_PATH.exists():
        raise SystemExit(
            f"输出文件已存在，为避免覆盖而停止：{OUTPUT_PATH}"
        )

    satellite = read_rgb(SAT_DIR, SATELLITE_NAME)
    uav = read_rgb(UAV_DIR, UAV_NAME)

    sat_gray = cv2.cvtColor(satellite, cv2.COLOR_RGB2GRAY)
    uav_gray = cv2.cvtColor(uav, cv2.COLOR_RGB2GRAY)

    sift = cv2.SIFT_create()
    sat_keypoints, sat_descriptors = sift.detectAndCompute(
        sat_gray, None
    )
    uav_keypoints, uav_descriptors = sift.detectAndCompute(
        uav_gray, None
    )

    if (
        sat_descriptors is None
        or uav_descriptors is None
        or len(uav_descriptors) < 2
    ):
        raise SystemExit("特征描述不足，无法进行留出试验。")

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    nearest_pairs = matcher.knnMatch(
        sat_descriptors,
        uav_descriptors,
        k=2,
    )

    ratio_matches = []

    for neighbors in nearest_pairs:
        if len(neighbors) < 2:
            continue

        best, second = neighbors

        if best.distance < RATIO_THRESHOLD * second.distance:
            ratio_matches.append(best)

    # 按描述符距离从小到大，确保每个无人机特征点只保留一次。
    # 这是减少重复对应的处理，不保证候选匹配一定正确。
    unique_matches = []
    used_uav_indices = set()

    for match in sorted(ratio_matches, key=lambda item: item.distance):
        if match.trainIdx in used_uav_indices:
            continue

        used_uav_indices.add(match.trainIdx)
        unique_matches.append(match)

    print(f"比例筛选后的候选：{len(ratio_matches)} 对")
    print(f"限制无人机特征重复后的候选：{len(unique_matches)} 对")

    if len(unique_matches) < 2 * MIN_TRAIN_POINTS:
        raise SystemExit("候选点不足，无法进行双向留出检查。")

    satellite_points = np.float32([
        sat_keypoints[match.queryIdx].pt
        for match in unique_matches
    ])

    uav_points = np.float32([
        uav_keypoints[match.trainIdx].pt
        for match in unique_matches
    ])

    # 用卫星图上候选点的 x 坐标中位数作为左右分界
    split_x = float(np.median(satellite_points[:, 0]))
    left = satellite_points[:, 0] < split_x
    right = ~left

    print(f"卫星图左右分界 x 坐标：{split_x:.2f}")
    print(f"左侧候选：{int(left.sum())} 对")
    print(f"右侧候选：{int(right.sum())} 对")

    results = [
        evaluate_split(
            "左侧拟合，右侧留出",
            satellite_points,
            uav_points,
            left,
            right,
        ),
        evaluate_split(
            "右侧拟合，左侧留出",
            satellite_points,
            uav_points,
            right,
            left,
        ),
    ]

    report = {
        "satellite_image": SATELLITE_NAME,
        "uav_image": UAV_NAME,
        "ratio_threshold": RATIO_THRESHOLD,
        "ransac_threshold_uav_pixels": RANSAC_THRESHOLD,
        "random_seed": RANDOM_SEED,
        "ratio_matches": len(ratio_matches),
        "unique_target_matches": len(unique_matches),
        "split_x_satellite_pixels": split_x,
        "results": results,
        "note": (
            "Held-out points are automatic SIFT candidates, not "
            "manually verified ground truth. Residuals may include "
            "false matches and are not independent absolute accuracy."
        ),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print(f"\n试验记录已保存：{OUTPUT_PATH}")
    print(
        "注意：留出点可能有错误匹配，"
        "上述数值不是人工核验后的真实配准精度。"
    )


if __name__ == "__main__":
    main()