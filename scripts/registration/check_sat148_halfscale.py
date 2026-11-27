import json
from pathlib import Path

import cv2
import numpy as np

root = Path.cwd()
directory = root / "data/interim/registration/mosaics_v1"
out = root / "data/interim/registration/sat148_halfscale_diagnostic_v1.json"

if out.exists():
    raise SystemExit(f"结果已存在，未覆盖：{out}")

cv2.setNumThreads(2)
cv2.setRNGSeed(42)

def load(name):
    with np.load(directory / name, allow_pickle=False) as data:
        return data["gray"].copy(), data["coverage"].astype(np.uint8)

def features(image, coverage):
    mask = cv2.erode(
        coverage * 255, np.ones((9, 9), dtype=np.uint8)
    )
    image = cv2.createCLAHE(
        clipLimit=2.0, tileGridSize=(16, 16)
    ).apply(image)
    return cv2.SIFT_create(
        nfeatures=40000, contrastThreshold=0.02
    ).detectAndCompute(image, mask)

def matches(a, b):
    matcher = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5), dict(checks=128)
    )
    result = {}
    for pair in matcher.knnMatch(a, b, k=2):
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance:
            result[pair[0].queryIdx] = pair[0].trainIdx
    return result

sat, sat_mask = load("satellite_component_148.npz")
uav, uav_mask = load("uav_component_2.npz")

sat = cv2.resize(
    sat, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA
)
sat_mask = cv2.resize(
    sat_mask, (sat.shape[1], sat.shape[0]),
    interpolation=cv2.INTER_NEAREST,
)

print("提取卫星特征……", flush=True)
sk, sd = features(sat, sat_mask)
print("提取无人机特征……", flush=True)
uk, ud = features(uav, uav_mask)

if sd is None or ud is None:
    raise SystemExit("未提取到描述子，请把提示发给我。")

forward = matches(sd, ud)
backward = matches(ud, sd)
pairs = []
seen_s, seen_u = set(), set()
for i, j in forward.items():
    if backward.get(j) != i:
        continue
    a = tuple(round(float(x), 1) for x in sk[i].pt)
    b = tuple(round(float(x), 1) for x in uk[j].pt)
    if a in seen_s or b in seen_u:
        continue
    seen_s.add(a)
    seen_u.add(b)
    pairs.append((i, j))

result = {
    "component": "148",
    "status": "diagnostic_only_not_approved",
    "satellite_features": len(sk),
    "uav_features": len(uk),
    "unique_mutual_matches": len(pairs),
    "satellite_additional_resize": 0.5,
}

if len(pairs) < 12:
    result["reason"] = "insufficient_matches_for_fit_and_holdout"
else:
    source = np.float32([sk[i].pt for i, j in pairs])
    target = np.float32([uk[j].pt for i, j in pairs])
    order = np.random.default_rng(42).permutation(len(pairs))
    holdout = order[::3]
    fit = np.setdiff1d(order, holdout)

    matrix, flags = cv2.estimateAffinePartial2D(
        source[fit], target[fit],
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=10000,
        confidence=0.999,
        refineIters=10,
    )

    result["fit_matches"] = len(fit)
    result["holdout_matches"] = len(holdout)

    if matrix is None:
        result["reason"] = "transform_not_estimated"
    else:
        inside = flags.ravel().astype(bool)
        errors = np.linalg.norm(
            source @ matrix[:, :2].T + matrix[:, 2] - target,
            axis=1,
        )
        original_matrix = matrix.copy()
        original_matrix[:, :2] *= 0.5
        scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))

        result.update({
            "matrix_original_sat_working_to_uav_working":
                original_matrix.tolist(),
            "scale_after_additional_resize": scale,
            "original_working_scale": scale * 0.5,
            "native_sat_to_uav_scale": scale,
            "rotation_degrees": float(np.degrees(
                np.arctan2(matrix[1, 0], matrix[0, 0])
            )),
            "fit_inliers": int(inside.sum()),
            "fit_span_resized_sat_xy": (
                np.ptp(source[fit][inside], axis=0).tolist()
                if inside.any() else None
            ),
            "holdout_within_3_working_pixels":
                int((errors[holdout] <= 3).sum()),
            "holdout_median_error_working_pixels":
                float(np.median(errors[holdout])),
            "limitation":
                "Descriptor-match holdout is not independent manual validation.",
        })

with out.open("x", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    f.write("\n")

print(json.dumps(result, ensure_ascii=False, indent=2))
print("\n保存：", out)
