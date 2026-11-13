"""
为 UAV 第 2 连通组的三个训练候选锚点寻找卫星影像线索。

结果仅供人工复核：
- 特征点匹配失败不能证明影像不重叠；
- RANSAC 内点多也不能直接证明同址；
- 暂不将结果用于自动划分或删除样本。
"""

import csv
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
from rasterio.errors import NotGeoreferencedWarning

from match_sift_pair import read_rgb, SAT_DIR, UAV_DIR


PROJECT = Path(__file__).resolve().parents[2]
OUTPUT = (
    PROJECT / "data/interim/registration/"
    "uav_group2_train_anchor_screening_v1.csv"
)

ANCHORS = [
    "luding_UAV0177.tif",
    "luding_UAV0354.tif",
    "luding_UAV0519.tif",
]

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
cv2.setNumThreads(1)


def get_features(folder, filename, sift):
    rgb = read_rgb(folder, filename)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return sift.detectAndCompute(gray, None)


def screen_pair(sat_name, sat_kp, sat_desc, uav_name, uav_kp, uav_desc):
    result = {
        "satellite": sat_name,
        "uav_image": uav_name,
        "ratio_matches": 0,
        "unique_target_matches": 0,
        "ransac_inliers": 0,
        "scale": "",
        "sat_inlier_span_x": "",
        "sat_inlier_span_y": "",
        "uav_inlier_span_x": "",
        "uav_inlier_span_y": "",
        "status": "insufficient_matches",
    }

    if sat_desc is None or uav_desc is None or len(uav_desc) < 2:
        return result

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(sat_desc, uav_desc, k=2)

    ratio_matches = [
        pair[0]
        for pair in pairs
        if len(pair) == 2
        and pair[0].distance < 0.75 * pair[1].distance
    ]
    result["ratio_matches"] = len(ratio_matches)

    # 同一个 UAV 描述子只保留距离最小的卫星匹配。
    best_by_target = {}
    for match in sorted(ratio_matches, key=lambda item: item.distance):
        if match.trainIdx not in best_by_target:
            best_by_target[match.trainIdx] = match

    matches = list(best_by_target.values())
    result["unique_target_matches"] = len(matches)

    if len(matches) < 4:
        return result

    source = np.float32(
        [sat_kp[match.queryIdx].pt for match in matches]
    ).reshape(-1, 1, 2)

    target = np.float32(
        [uav_kp[match.trainIdx].pt for match in matches]
    ).reshape(-1, 1, 2)

    cv2.setRNGSeed(42)
    matrix, mask = cv2.estimateAffinePartial2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=5000,
        confidence=0.99,
    )

    if matrix is None or mask is None:
        result["status"] = "no_transform"
        return result

    flags = mask.ravel().astype(bool)
    inlier_count = int(flags.sum())
    scale = float(np.hypot(matrix[0, 0], matrix[0, 1]))

    result["ransac_inliers"] = inlier_count

    if np.isfinite(scale):
        result["scale"] = scale

    if (
        not np.isfinite(matrix).all()
        or not np.isfinite(scale)
        or not (1e-3 <= scale <= 1e3)
    ):
        result["status"] = "degenerate_transform"
        return result

    if inlier_count < 4:
        result["status"] = "fewer_than_4_inliers"
        return result

    source_inliers = source.reshape(-1, 2)[flags]
    target_inliers = target.reshape(-1, 2)[flags]

    source_span = np.ptp(source_inliers, axis=0)
    target_span = np.ptp(target_inliers, axis=0)

    result["sat_inlier_span_x"] = float(source_span[0])
    result["sat_inlier_span_y"] = float(source_span[1])
    result["uav_inlier_span_x"] = float(target_span[0])
    result["uav_inlier_span_y"] = float(target_span[1])
    result["status"] = "candidate_requires_review"

    return result


def main():
    if OUTPUT.exists():
        raise SystemExit(f"结果文件已存在，未覆盖：{OUTPUT}")

    sat_files = sorted(SAT_DIR.glob("*.tif"))

    if len(sat_files) != 652:
        raise SystemExit(f"卫星影像数量异常：{len(sat_files)}")

    sift = cv2.SIFT_create()
    anchor_features = {}

    for name in ANCHORS:
        anchor_features[name] = get_features(UAV_DIR, name, sift)
        print(f"锚点特征：{name}，{len(anchor_features[name][0])} 个")

    results = []
    errors = []

    print("\n开始筛查 652 张卫星切片……", flush=True)

    for index, path in enumerate(sat_files, start=1):
        try:
            sat_kp, sat_desc = get_features(SAT_DIR, path.name, sift)
            file_results = []

            for anchor in ANCHORS:
                uav_kp, uav_desc = anchor_features[anchor]

                file_results.append(
                    screen_pair(
                        path.name, sat_kp, sat_desc,
                        anchor, uav_kp, uav_desc,
                    )
                )

            results.extend(file_results)

        except Exception as error:
            errors.append((path.name, str(error)))

        if index % 50 == 0 or index == len(sat_files):
            print(
                f"  已处理 {index}/{len(sat_files)}，"
                f"失败 {len(errors)} 张",
                flush=True,
            )

    if errors:
        for name, message in errors[:10]:
            print(f"失败：{name}：{message}")
        raise SystemExit("存在失败记录，未保存筛查表。")

    expected = len(sat_files) * len(ANCHORS)
    if len(results) != expected:
        raise SystemExit(
            f"结果数量异常：{len(results)}，预期 {expected}；未保存。"
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    for anchor in ANCHORS:
        candidates = [
            row for row in results
            if row["uav_image"] == anchor
            and row["status"] == "candidate_requires_review"
        ]

        candidates.sort(
            key=lambda row: row["ransac_inliers"],
            reverse=True,
        )

        print(f"\n{anchor}：内点数最高的 5 个待复核候选")

        if not candidates:
            print("  没有通过基本检查的候选。")

        for row in candidates[:5]:
            print(
                f"  {row['satellite']}："
                f"内点={row['ransac_inliers']}，"
                f"缩放={row['scale']:.3f}，"
                f"卫星点范围="
                f"{row['sat_inlier_span_x']:.1f}×"
                f"{row['sat_inlier_span_y']:.1f} 像素，"
                f"UAV 点范围="
                f"{row['uav_inlier_span_x']:.1f}×"
                f"{row['uav_inlier_span_y']:.1f} 像素"
            )

    print(f"\n统计表已保存：{OUTPUT}")
    print("候选均需进一步核查；未匹配不代表没有空间重叠。")


if __name__ == "__main__":
    main()
