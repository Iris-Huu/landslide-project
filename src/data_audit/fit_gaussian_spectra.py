"""
复现 Moxitaidi 卫星与 UAV 局部窗口的高斯频谱拟合。

使用：
    python src/data_audit/fit_gaussian_spectra.py

脚本重新计算已有的两组窗口，并与原 JSON 中的 sigma 和差异分数比较。
它只读取已有文件，不修改原始影像或已有结果。

限制：这是单张卫星影像上的探索性相对模糊试验，不是绝对有效
分辨率测量。插值、有效区域和跨平台成像差异均会影响结果。
"""

import json
from pathlib import Path

import cv2
import numpy as np

from compare_sample_spectra import read_rgb, radial_power


PROJECT = Path(__file__).resolve().parents[2]

SAT_PATH = (
    PROJECT / "data/raw/CAS/Moxitaidi_SAT/img/"
    "Moxitaidi_Satellite001.tif"
)
UAV_DIR = PROJECT / "data/raw/CAS/Moxitaidi_UAV_0.6m/img"
REGISTRATION_DIR = PROJECT / "data/interim/registration"

CASES = [
    {
        "uav": "luding_UAV0011.tif",
        "transform": "sat001_uav0011_min_inliers_check_result.json",
        "original_result": "sat001_uav0011_gaussian_fit_v1.json",
        "windows": [(0, 0), (0, 256)],
    },
    {
        "uav": "luding_UAV0001.tif",
        "transform": "sat001_uav0001_min_inliers_check_result.json",
        "original_result": "sat001_uav0001_gaussian_fit_v1.json",
        "windows": [(256, 0), (256, 256)],
    },
]

SIGMAS = np.arange(0.0, 4.01, 0.1)
WINDOW_SIZE = 256


def load_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def fit_window(sat_gray, uav_gray, row, col):
    window = (
        slice(row, row + WINDOW_SIZE),
        slice(col, col + WINDOW_SIZE),
    )

    frequency, sat_power = radial_power(sat_gray[window])

    low = (frequency >= 0.02) & (frequency < 0.08)
    fit_band = (frequency >= 0.04) & (frequency < 0.20)

    sat_reference = float(np.nanmean(sat_power[low]))

    if not np.isfinite(sat_reference) or sat_reference <= 0:
        raise ValueError(f"窗口 ({row}, {col}) 的卫星参考功率异常")

    sat_normalized = sat_power / sat_reference
    scores = []

    for sigma in SIGMAS:
        if sigma == 0:
            blurred = uav_gray
        else:
            blurred = cv2.GaussianBlur(
                uav_gray,
                (0, 0),
                sigmaX=float(sigma),
                sigmaY=float(sigma),
            )

        _, uav_power = radial_power(blurred[window])
        uav_reference = float(np.nanmean(uav_power[low]))

        if not np.isfinite(uav_reference) or uav_reference <= 0:
            scores.append(float("inf"))
            continue

        uav_normalized = uav_power / uav_reference

        valid = (
            fit_band
            & np.isfinite(sat_normalized)
            & np.isfinite(uav_normalized)
            & (sat_normalized > 0)
            & (uav_normalized > 0)
        )

        if not valid.any():
            scores.append(float("inf"))
            continue

        difference = (
            np.log(sat_normalized[valid])
            - np.log(uav_normalized[valid])
        )
        scores.append(float(np.mean(difference ** 2)))

    best = int(np.argmin(scores))

    if not np.isfinite(scores[best]) or not np.isfinite(scores[0]):
        raise ValueError(f"窗口 ({row}, {col}) 无有效拟合结果")

    return {
        "row_start": row,
        "col_start": col,
        "best_sigma_uav_pixels": float(SIGMAS[best]),
        "score_without_blur": scores[0],
        "best_score": scores[best],
        "at_search_boundary": best in (0, len(SIGMAS) - 1),
    }


def window_key(item):
    # 兼容两次原始记录使用的字段名。
    row = item.get("window_row_start", item.get("row_start"))
    col = item.get("window_col_start", item.get("col_start"))

    if row is None or col is None:
        raise ValueError(f"原始结果缺少窗口坐标：{item}")

    return int(row), int(col)


def run_case(case):
    transform_path = REGISTRATION_DIR / case["transform"]
    original_path = REGISTRATION_DIR / case["original_result"]

    transform_record = load_json(transform_path)
    original_record = load_json(original_path)

    if (
        original_record.get("satellite") != SAT_PATH.name
        or original_record.get("uav") != case["uav"]
    ):
        raise ValueError(f"原始结果中的影像名称不匹配：{original_path}")

    matrix = np.asarray(
        transform_record["transform_satellite_to_uav"],
        dtype=np.float64,
    )

    if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"变换矩阵无效：{transform_path}")

    satellite = read_rgb(SAT_PATH)
    uav = read_rgb(UAV_DIR / case["uav"])

    warped_sat = cv2.warpAffine(
        satellite,
        matrix,
        (512, 512),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    coverage = cv2.warpAffine(
        np.ones((512, 512), dtype=np.uint8),
        matrix,
        (512, 512),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    sat_gray = cv2.cvtColor(warped_sat, cv2.COLOR_RGB2GRAY)
    uav_gray = cv2.cvtColor(uav, cv2.COLOR_RGB2GRAY).astype(np.float32)

    original_items = original_record["results"]
    originals = {window_key(item): item for item in original_items}

    if len(originals) != len(original_items):
        raise ValueError(f"原始结果含重复窗口：{original_path}")

    if set(originals) != set(case["windows"]):
        raise ValueError(f"原始结果与预期窗口不一致：{original_path}")

    print(f"\n影像对：{SAT_PATH.name} → {case['uav']}")
    all_match = True

    for row, col in case["windows"]:
        window = (
            slice(row, row + WINDOW_SIZE),
            slice(col, col + WINDOW_SIZE),
        )

        covered = float(coverage[window].mean())

        if covered < 1.0:
            raise ValueError(
                f"窗口 ({row}, {col}) 几何覆盖不足：{covered:.2%}"
            )

        current = fit_window(sat_gray, uav_gray, row, col)
        original = originals[(row, col)]

        fields = [
            "best_sigma_uav_pixels",
            "score_without_blur",
            "best_score",
        ]

        matches = all(
            np.isclose(
                current[field],
                float(original[field]),
                rtol=1e-6,
                atol=1e-8,
            )
            for field in fields
        )

        matches = matches and (
            current["at_search_boundary"]
            == original["at_search_boundary"]
        )

        all_match = all_match and matches

        print(
            f"  窗口 ({row}, {col})："
            f"sigma={current['best_sigma_uav_pixels']:.2f}，"
            f"原始差异={current['score_without_blur']:.4f}，"
            f"最佳差异={current['best_score']:.4f}，"
            f"与原记录{'一致' if matches else '不一致'}"
        )

    return all_match


def main():
    outcomes = [run_case(case) for case in CASES]

    if not all(outcomes):
        raise SystemExit(
            "\n存在与原记录不一致的窗口，请检查环境和输入文件。"
        )

    print("\n4 个窗口均与原记录一致。")
    print("注意：这验证的是计算复现，不证明有效分辨率已测定。")


if __name__ == "__main__":
    main()
