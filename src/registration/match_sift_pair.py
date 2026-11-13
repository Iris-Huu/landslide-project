"""
对指定的 Moxitaidi 卫星与无人机切片进行 SIFT 匹配。

用法示例：
python src/registration/match_sift_pair.py \
  --sat Moxitaidi_Satellite001.tif \
  --uav luding_UAV0011.tif \
  --name sat001_uav0011

注意：
- 本程序用于探索性匹配；
- 不使用尚未验证的无数据区掩码；
- RANSAC 内点误差不是独立配准精度；
- 匹配失败不代表两套数据不存在同址区域。
"""

from pathlib import Path
import argparse
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

SAT_DIR = (
    PROJECT_DIR / "data/raw/CAS/Moxitaidi_SAT/img"
)
UAV_DIR = (
    PROJECT_DIR / "data/raw/CAS/Moxitaidi_UAV_0.6m/img"
)
FIGURE_DIR = PROJECT_DIR / "reports/figures"
RESULT_DIR = PROJECT_DIR / "data/interim/registration"

RATIO_THRESHOLD = 0.75
RANSAC_THRESHOLD = 3.0
MAX_LINES = 100
RANDOM_SEED = 42

warnings.filterwarnings(
    "ignore",
    category=NotGeoreferencedWarning,
)


def read_rgb(folder, filename):
    """读取一张指定目录中的 RGB 影像。"""
    # 只接受文件名，避免意外指定到其他目录
    if Path(filename).name != filename:
        raise ValueError(f"请输入文件名，不要输入路径：{filename}")

    path = folder / filename

    if not path.is_file():
        raise FileNotFoundError(f"没有找到影像：{path}")

    with rasterio.open(path) as src:
        image = src.read()

    if image.shape != (3, 512, 512):
        raise ValueError(f"{filename} 的形状异常：{image.shape}")

    if image.dtype != np.uint8:
        raise ValueError(f"{filename} 不是 uint8：{image.dtype}")

    return np.moveaxis(image, 0, -1).copy()


def parse_args():
    parser = argparse.ArgumentParser(
        description="匹配一对 Moxitaidi 卫星与无人机影像。"
    )

    parser.add_argument(
        "--sat",
        required=True,
        help="卫星影像文件名",
    )
    parser.add_argument(
        "--uav",
        required=True,
        help="无人机影像文件名",
    )
    parser.add_argument(
        "--name",
        required=True,
        help="本次结果名称，只能使用字母、数字、下划线或连字符",
    )

    args = parser.parse_args()

    if not args.name or not all(
        char.isascii() and (char.isalnum() or char in "_-")
        for char in args.name
    ):
        parser.error(
            "--name 只能包含英文字母、数字、下划线或连字符"
        )

    return args


def main():
    args = parse_args()

    figure_path = FIGURE_DIR / f"{args.name}_matches.png"
    result_path = RESULT_DIR / f"{args.name}_result.json"

    for path in [figure_path, result_path]:
        if path.exists():
            raise SystemExit(
                f"输出文件已存在，为避免覆盖而停止：{path}\n"
                "如需再次运行，请使用不同的 --name。"
            )

    satellite = read_rgb(SAT_DIR, args.sat)
    uav = read_rgb(UAV_DIR, args.uav)

    sat_gray = cv2.cvtColor(
        satellite, cv2.COLOR_RGB2GRAY
    )
    uav_gray = cv2.cvtColor(
        uav, cv2.COLOR_RGB2GRAY
    )

    sift = cv2.SIFT_create()

    sat_points, sat_descriptors = sift.detectAndCompute(
        sat_gray, None
    )
    uav_points, uav_descriptors = sift.detectAndCompute(
        uav_gray, None
    )

    result = {
        "satellite_image": args.sat,
        "uav_image": args.uav,
        "status": "not_estimated",
        "ratio_threshold": RATIO_THRESHOLD,
        "ransac_threshold_uav_pixels": RANSAC_THRESHOLD,
        "random_seed": RANDOM_SEED,
        "satellite_keypoints": len(sat_points),
        "uav_keypoints": len(uav_points),
        "candidate_matches": 0,
        "ransac_inliers": 0,
        "transform_satellite_to_uav": None,
        "median_inlier_error_uav_pixels": None,
        "note": (
            "Exploratory registration. Inlier error is not an "
            "independent accuracy assessment."
        ),
    }

    print(f"卫星影像：{args.sat}")
    print(f"无人机影像：{args.uav}")
    print(f"卫星特征点：{len(sat_points)}")
    print(f"无人机特征点：{len(uav_points)}")

    candidates = []
    inliers = []
    matrix = None

    if (
        sat_descriptors is not None
        and uav_descriptors is not None
        and len(uav_descriptors) >= 2
    ):
        matcher = cv2.BFMatcher(cv2.NORM_L2)

        nearest_pairs = matcher.knnMatch(
            sat_descriptors,
            uav_descriptors,
            k=2,
        )

        for neighbors in nearest_pairs:
            if len(neighbors) < 2:
                continue

            best, second = neighbors

            if best.distance < RATIO_THRESHOLD * second.distance:
                candidates.append(best)

    result["candidate_matches"] = len(candidates)
    print(f"比例筛选后的候选匹配：{len(candidates)}")

    if len(candidates) >= 4:
        satellite_xy = np.float32([
            sat_points[match.queryIdx].pt
            for match in candidates
        ]).reshape(-1, 1, 2)

        uav_xy = np.float32([
            uav_points[match.trainIdx].pt
            for match in candidates
        ]).reshape(-1, 1, 2)

        cv2.setRNGSeed(RANDOM_SEED)

        matrix, inlier_mask = cv2.estimateAffinePartial2D(
            satellite_xy,
            uav_xy,
            method=cv2.RANSAC,
            ransacReprojThreshold=RANSAC_THRESHOLD,
            maxIters=5000,
            confidence=0.99,
        )

        if matrix is not None and inlier_mask is not None:
            flags = inlier_mask.ravel().astype(bool)

            inliers = [
                match
                for match, keep in zip(candidates, flags)
                if keep
            ]

            result["ransac_inliers"] = len(inliers)
            result["transform_satellite_to_uav"] = matrix.tolist()
            result["status"] = "model_estimated_requires_review"

            if inliers:
                source = satellite_xy.reshape(-1, 2)[flags]
                target = uav_xy.reshape(-1, 2)[flags]

                predicted = source @ matrix[:, :2].T + matrix[:, 2]
                residuals = np.linalg.norm(
                    predicted - target,
                    axis=1,
                )

                result["median_inlier_error_uav_pixels"] = float(
                    np.median(residuals)
                )

    print(f"RANSAC 几何一致的匹配：{len(inliers)}")

    if matrix is not None:
        print("\n相似变换矩阵（卫星像素 → 无人机像素）：")
        print(matrix)

        error = result["median_inlier_error_uav_pixels"]
        if error is not None:
            print(f"内点误差中位数：{error:.2f} 像素")
    else:
        print("本次未能估计出变换；请记录并检查样本覆盖范围。")

    # 生成对照图：如果无内点，则只并排展示原图
    shown = sorted(
        inliers,
        key=lambda match: match.distance,
    )[:MAX_LINES]

    sat_bgr = cv2.cvtColor(satellite, cv2.COLOR_RGB2BGR)
    uav_bgr = cv2.cvtColor(uav, cv2.COLOR_RGB2BGR)

    preview_bgr = cv2.drawMatches(
        sat_bgr,
        sat_points,
        uav_bgr,
        uav_points,
        shown,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    preview_rgb = cv2.cvtColor(
        preview_bgr, cv2.COLOR_BGR2RGB
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(preview_rgb)
    ax.set_title(
        f"{args.sat} → {args.uav}: "
        f"{len(inliers)} RANSAC inliers "
        f"(showing {len(shown)})"
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    with result_path.open(
        "x", encoding="utf-8"
    ) as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    print(f"\n匹配图：{figure_path}")
    print(f"统计记录：{result_path}")
    print(
        "提示：请人工检查匹配分布和地物对应关系；"
        "不要仅凭匹配数量判定配准质量。"
    )


if __name__ == "__main__":
    main()