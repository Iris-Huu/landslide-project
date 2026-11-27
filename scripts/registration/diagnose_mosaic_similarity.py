import json
import math
from pathlib import Path

import cv2
import numpy as np
import rasterio

ROOT = Path.cwd()
DIR = ROOT / "data/interim/registration/mosaics_v1"
OUT = ROOT / "data/interim/registration/mosaic_similarity_diagnostic_v1.json"

if OUT.exists():
    raise SystemExit(f"结果已存在，未覆盖：{OUT}")

cv2.setNumThreads(2)
cv2.setRNGSeed(42)

def load_mosaic(name):
    with np.load(DIR / name, allow_pickle=False) as data:
        return data["gray"].copy(), data["coverage"].astype(bool)

def features(gray, coverage):
    mask = cv2.erode(
        coverage.astype(np.uint8) * 255,
        np.ones((9, 9), np.uint8),
    )
    enhanced = cv2.createCLAHE(
        clipLimit=2.0, tileGridSize=(16, 16)
    ).apply(gray.astype(np.uint8))
    sift = cv2.SIFT_create(nfeatures=40000, contrastThreshold=0.02)
    return sift.detectAndCompute(enhanced, mask)

def ratio_matches(matcher, a, b):
    result = {}
    for pair in matcher.knnMatch(a, b, k=2):
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
            result[pair[0].queryIdx] = pair[0].trainIdx
    return result

uav_gray, uav_mask = load_mosaic("uav_component_2.npz")
print("提取 UAV 拼图特征……", flush=True)
uav_kp, uav_desc = features(uav_gray, uav_mask)
print("UAV 特征数：", len(uav_kp), flush=True)

with rasterio.open(
    ROOT / "data/raw/CAS/Moxitaidi_SAT/img/Moxitaidi_Satellite100.tif"
) as ds:
    rgb = np.moveaxis(ds.read(), 0, -1)
gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
anchor = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)

tasks = [
    ("sat100_positive_control", anchor, np.ones(anchor.shape, dtype=bool)),
]
for component in ("154", "158", "148"):
    image, coverage = load_mosaic(f"satellite_component_{component}.npz")
    tasks.append((f"satellite_component_{component}", image, coverage))

results = []
for name, image, coverage in tasks:
    print(f"\n处理 {name}……", flush=True)
    kp, desc = features(image, coverage)
    item = {
        "name": name,
        "satellite_features": len(kp),
        "status": "diagnostic_only_not_approved",
    }
    if desc is None or uav_desc is None:
        item["reason"] = "missing_descriptors"
        results.append(item)
        print(item, flush=True)
        continue

    matcher = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5), dict(checks=128)
    )
    forward = ratio_matches(matcher, desc, uav_desc)
    backward = ratio_matches(matcher, uav_desc, desc)
    pairs = [
        (i, j) for i, j in forward.items()
        if backward.get(j) == i
    ]

    # 去除同一空间点因不同 SIFT 方向重复出现的匹配。
    unique = []
    seen_source, seen_target = set(), set()
    for i, j in pairs:
        source = tuple(round(float(v), 1) for v in kp[i].pt)
        target = tuple(round(float(v), 1) for v in uav_kp[j].pt)
        if source in seen_source or target in seen_target:
            continue
        seen_source.add(source)
        seen_target.add(target)
        unique.append((i, j))

    item["unique_mutual_matches"] = len(unique)
    if len(unique) < 12:
        item["reason"] = "insufficient_matches_for_fit_and_holdout"
    else:
        source = np.float32([kp[i].pt for i, j in unique])
        target = np.float32([uav_kp[j].pt for i, j in unique])
        order = np.random.default_rng(42).permutation(len(unique))
        holdout = order[::3]
        fit = np.setdiff1d(order, holdout)

        matrix, inliers = cv2.estimateAffinePartial2D(
            source[fit], target[fit],
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=10000,
            confidence=0.999,
            refineIters=10,
        )
        item["fit_matches"] = len(fit)
        item["holdout_matches"] = len(holdout)

        if matrix is None:
            item["reason"] = "transform_not_estimated"
        else:
            inside = inliers.ravel().astype(bool)
            predicted = source @ matrix[:, :2].T + matrix[:, 2]
            errors = np.linalg.norm(predicted - target, axis=1)
            supported = errors[holdout] <= 3.0
            item.update({
                "matrix_sat_working_to_uav_working": matrix.tolist(),
                "working_scale": float(math.hypot(matrix[0, 0], matrix[1, 0])),
                "rotation_degrees": float(math.degrees(
                    math.atan2(matrix[1, 0], matrix[0, 0])
                )),
                "fit_inliers": int(inside.sum()),
                "fit_span_sat_working_xy": (
                    np.ptp(source[fit][inside], axis=0).tolist()
                    if inside.any() else None
                ),
                "holdout_within_3_working_pixels": int(supported.sum()),
                "holdout_median_error_working_pixels": float(
                    np.median(errors[holdout])
                ),
                "limitation": (
                    "Held-out descriptor matches are a diagnostic, "
                    "not independent manual registration validation."
                ),
            })

    results.append(item)
    print(json.dumps(item, ensure_ascii=False, indent=2), flush=True)

with OUT.open("x", encoding="utf-8") as f:
    json.dump({
        "scope": "registration diagnostics only; no labels or model outputs",
        "expected_working_scale_under_native_ratio_2": 1.0,
        "results": results,
    }, f, ensure_ascii=False, indent=2, allow_nan=False)
    f.write("\n")

print("\n保存：", OUT)
