import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "data/interim/registration/mosaics_v1"
OUT = ROOT / "data/interim/registration/mosaic_translation_candidates_v1.json"
cv2.setNumThreads(2)
cv2.setRNGSeed(42)

if OUT.exists():
    raise SystemExit(f"结果已存在，未覆盖：{OUT}")

def load_features(path, limit):
    with np.load(path, allow_pickle=False) as data:
        gray = data["gray"]
        coverage = data["coverage"]
        metadata = json.loads(str(data["metadata"].item()))

    mask = cv2.erode(
        coverage.astype(np.uint8),
        np.ones((9, 9), dtype=np.uint8),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) * 255

    enhanced = cv2.createCLAHE(
        clipLimit=2.0, tileGridSize=(16, 16)
    ).apply(gray)

    sift = cv2.SIFT_create(nfeatures=limit)
    keypoints, descriptors = sift.detectAndCompute(enhanced, mask)

    points = np.array([k.pt for k in keypoints], dtype=np.float32).reshape(-1, 2)
    return points, descriptors, metadata

def ratio_matches(a, b):
    matcher = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5),
        dict(checks=96),
    )
    accepted = {}
    for pair in matcher.knnMatch(a, b, k=2):
        if len(pair) == 2:
            first, second = pair
            if first.distance < 0.70 * second.distance:
                accepted[first.queryIdx] = first.trainIdx
    return accepted

def estimate_translation(delta):
    # 固定尺度1、旋转0；这里只拟合工作拼图上的平移。
    bins = np.floor(delta / 6.0).astype(np.int64)
    peaks = Counter(map(tuple, bins)).most_common(30)

    best_mask = None
    best_shift = None
    best_score = (-1, float("-inf"))

    for cell, _ in peaks:
        center = (np.array(cell, dtype=float) + 0.5) * 6.0
        near = np.linalg.norm(delta - center, axis=1) <= 12.0
        if not near.any():
            continue

        shift = np.median(delta[near], axis=0)
        for _ in range(3):
            residual = np.linalg.norm(delta - shift, axis=1)
            inliers = residual <= 3.0
            if not inliers.any():
                break
            shift = np.median(delta[inliers], axis=0)

        residual = np.linalg.norm(delta - shift, axis=1)
        inliers = residual <= 3.0
        if not inliers.any():
            continue

        score = (int(inliers.sum()), -float(np.median(residual[inliers])))
        if score > best_score:
            best_score = score
            best_shift = shift
            best_mask = inliers

    return best_shift, best_mask

print("提取 UAV 拼图特征……", flush=True)
uav_points, uav_desc, uav_meta = load_features(
    DIR / "uav_component_2.npz", 20000
)
print("UAV 特征数：", len(uav_points), flush=True)

if uav_desc is None or len(uav_desc) < 2:
    raise SystemExit("UAV 特征不足。")

results = []

for component in ("154", "158", "148"):
    print(f"\n处理 SAT 连通组 {component}……", flush=True)
    sat_points, sat_desc, sat_meta = load_features(
        DIR / f"satellite_component_{component}.npz", 12000
    )

    result = {
        "sat_component": component,
        "sat_metadata": sat_meta,
        "sat_keypoints": len(sat_points),
        "status": "insufficient_matches",
    }

    if sat_desc is None or len(sat_desc) < 2:
        results.append(result)
        print(result["status"], flush=True)
        continue

    forward = ratio_matches(sat_desc, uav_desc)
    backward = ratio_matches(uav_desc, sat_desc)
    pairs = sorted(
        (i, j) for i, j in forward.items()
        if backward.get(j) == i
    )
    result["mutual_ratio_matches"] = len(pairs)
    print("双向一致匹配数：", len(pairs), flush=True)

    if len(pairs) < 6:
        results.append(result)
        continue

    sat_xy = np.array([sat_points[i] for i, j in pairs])
    uav_xy = np.array([uav_points[j] for i, j in pairs])
    delta = uav_xy - sat_xy

    rng = np.random.default_rng(42)
    order = rng.permutation(len(pairs))
    holdout = order[:max(2, len(pairs) // 3)]
    fitting = order[len(holdout):]

    shift, fit_inliers = estimate_translation(delta[fitting])
    if shift is None:
        result["status"] = "translation_not_found"
        results.append(result)
        continue

    residual = np.linalg.norm(delta - shift, axis=1)
    holdout_errors = residual[holdout]
    accepted_holdout = holdout_errors <= 3.0
    all_inliers = residual <= 3.0
    fit_xy = sat_xy[fitting][fit_inliers]

    result.update({
        "status": "candidate_requires_spatial_review",
        "working_transform": "uav_xy = satellite_xy + translation",
        "working_translation_xy": shift.tolist(),
        "working_pixel_in_uav_native_pixels": 4,
        "sat_native_to_uav_native_scale_assumption": 2,
        "fit_matches": len(fitting),
        "fit_inliers": int(fit_inliers.sum()),
        "fit_inlier_span_sat_working_xy": np.ptp(fit_xy, axis=0).tolist(),
        "holdout_matches": len(holdout),
        "holdout_within_3_working_pixels": int(accepted_holdout.sum()),
        "holdout_all_median_error_working_pixels": float(np.median(holdout_errors)),
        "holdout_supported_median_error_uav_pixels": (
            float(np.median(holdout_errors[accepted_holdout]) * 4)
            if accepted_holdout.any() else None
        ),
        "all_consistent_matches": int(all_inliers.sum()),
        "matches": [
            {
                "sat_xy": sat_xy[k].tolist(),
                "uav_xy": uav_xy[k].tolist(),
                "error_working_pixels": float(residual[k]),
                "partition": "holdout" if k in set(holdout.tolist()) else "fit",
            }
            for k in range(len(pairs))
        ],
    })
    results.append(result)

    for key in (
        "working_translation_xy", "fit_inliers",
        "fit_inlier_span_sat_working_xy",
        "holdout_matches", "holdout_within_3_working_pixels",
        "holdout_all_median_error_working_pixels",
        "holdout_supported_median_error_uav_pixels",
    ):
        print(f"{key}：{result[key]}", flush=True)

report = {
    "purpose": "CPU geometry-only translation candidates; not approved B samples",
    "uav_metadata": uav_meta,
    "uav_keypoints": len(uav_points),
    "method": "CLAHE SIFT, mutual ratio 0.70, fixed-scale translation",
    "fit_tolerance_working_pixels": 3,
    "limitations": [
        "尺度2和旋转0为固定工作假设。",
        "留出匹配点用于检查一致性，不是独立地理控制点。",
        "尚未进行空间覆盖、重复纹理和不同平移峰的人工复核。",
        "不能直接将候选变换用于正式B单元批准。",
    ],
    "results": results,
}

with OUT.open("x", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
    f.write("\n")

print("\n结果保存：", OUT, flush=True)
