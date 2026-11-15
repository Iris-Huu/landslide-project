"""
针对指定SAT切片与单张UAV切片开展探索性配准筛查。

复用 screen_satellite_anchors.py 中的 screen_pair。
只输出候选关系，不自动批准跨平台配准或训练使用。
"""

import argparse
import csv
from pathlib import Path

import cv2

from match_sift_pair import read_rgb, SAT_DIR, UAV_DIR
from screen_satellite_anchors import screen_pair


PROJECT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sat", nargs="+", required=True)
    parser.add_argument("--uav", required=True)
    parser.add_argument("--name", required=True)

    args = parser.parse_args()

    if not args.name or not all(
        char.isascii()
        and (char.isalnum() or char in "_-")
        for char in args.name
    ):
        parser.error("--name只能使用英文数字、下划线或连字符")

    return args


def get_features(folder, filename, sift):
    image = read_rgb(folder, filename)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return sift.detectAndCompute(gray, None)


def main():
    args = parse_args()

    output = (
        PROJECT / "data/interim/registration"
        / f"{args.name}_screening.csv"
    )

    if output.exists():
        raise SystemExit(f"输出已存在，拒绝覆盖：{output}")

    cv2.setNumThreads(1)
    sift = cv2.SIFT_create()

    uav_kp, uav_desc = get_features(
        UAV_DIR, args.uav, sift
    )

    results = []

    for sat_name in args.sat:
        sat_kp, sat_desc = get_features(
            SAT_DIR, sat_name, sift
        )

        result = screen_pair(
            sat_name, sat_kp, sat_desc,
            args.uav, uav_kp, uav_desc
        )

        result["satellite_keypoints"] = len(sat_kp)
        result["uav_keypoints"] = len(uav_kp)
        result["ratio_threshold"] = 0.75
        result["ransac_threshold_uav_pixels"] = 3.0
        result["random_seed"] = 42

        results.append(result)

    if not results:
        raise SystemExit("没有生成筛查结果，未保存文件。")

    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(results[0])
        )
        writer.writeheader()
        writer.writerows(results)

    print("===== 小规模配准筛查结果 =====")

    for row in results:
        print(
            row["satellite"],
            "| ratio:", row["ratio_matches"],
            "| unique:", row["unique_target_matches"],
            "| RANSAC:", row["ransac_inliers"],
            "| status:", row["status"]
        )

    print("\n结果文件：", output)
    print("注意：筛查结果不能证明地理重叠或无泄漏。")


if __name__ == "__main__":
    main()
