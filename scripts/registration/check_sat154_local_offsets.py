import json
from pathlib import Path

import cv2
import numpy as np

root = Path.cwd()
directory = root / "data/interim/registration/mosaics_v1"
selection_path = root / "reports/figures/sat154_local_review_v1/selection.json"
out = root / "reports/tables/sat154_local_offset_diagnostic_v1.json"

if out.exists():
    raise SystemExit(f"结果已存在，未覆盖：{out}")

cv2.setNumThreads(2)

def load(name):
    with np.load(directory / name, allow_pickle=False) as data:
        return data["gray"].copy(), data["coverage"].astype(np.uint8)

def gradient(image):
    image = cv2.GaussianBlur(image.astype(np.float32), (5, 5), 1.0)
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)

def correlate(image, template):
    return cv2.matchTemplate(
        np.ascontiguousarray(image, dtype=np.float32),
        np.ascontiguousarray(template, dtype=np.float32),
        cv2.TM_CCORR,
    )

def masked_ncc(search, search_mask, template, template_mask):
    # 每个平移位置只统计两幅图均有覆盖的像素。
    ma = template_mask.astype(np.float32)
    mb = search_mask.astype(np.float32)
    a = template * ma
    b = search * mb

    count = np.maximum(correlate(mb, ma), 0)
    safe = np.maximum(count, 1)
    sa = correlate(mb, a)
    sb = correlate(b, ma)
    saa = correlate(mb, template * a)
    sbb = correlate(search * b, ma)
    sab = correlate(b, a)

    covariance = sab - sa * sb / safe
    va = np.maximum(saa - sa * sa / safe, 0)
    vb = np.maximum(sbb - sb * sb / safe, 0)
    denominator = np.sqrt(va * vb)
    score = np.full(count.shape, -np.inf, dtype=np.float32)
    usable = (
        (count >= 0.80 * ma.sum())
        & (denominator > 1e-6)
    )
    score[usable] = covariance[usable] / denominator[usable]
    return score, count

report = json.loads(
    (root / "data/interim/registration/mosaic_similarity_diagnostic_v1.json")
    .read_text(encoding="utf-8")
)
entry = next(
    r for r in report["results"]
    if r["name"] == "satellite_component_154"
)
matrix = np.asarray(
    entry["matrix_sat_working_to_uav_working"], dtype=np.float64
)
selection = json.loads(selection_path.read_text(encoding="utf-8"))

sat, sat_mask = load("satellite_component_154.npz")
uav, uav_mask = load("uav_component_2.npz")
size = (uav.shape[1], uav.shape[0])

warped = cv2.warpAffine(sat, matrix, size, flags=cv2.INTER_LINEAR)
warped_mask = cv2.warpAffine(
    sat_mask, matrix, size, flags=cv2.INTER_NEAREST
)
kernel = np.ones((7, 7), dtype=np.uint8)
warped_mask = cv2.erode(warped_mask, kernel)
uav_mask = cv2.erode(uav_mask, kernel)

sat_gradient = gradient(warped)
uav_gradient = gradient(uav)

radius = 24
results = []

for location in selection["locations"]:
    x, y = location["center_uav_working_xy"]
    half = location["window_working_pixels"] // 2
    left, top = x - half, y - half
    right, bottom = x + half, y + half

    if (
        left < radius or top < radius
        or right + radius > uav.shape[1]
        or bottom + radius > uav.shape[0]
    ):
        raise ValueError("窗口距拼图边界过近，停止。")

    template = sat_gradient[top:bottom, left:right]
    template_mask = warped_mask[top:bottom, left:right]
    search = uav_gradient[
        top-radius:bottom+radius,
        left-radius:right+radius,
    ]
    search_mask = uav_mask[
        top-radius:bottom+radius,
        left-radius:right+radius,
    ]

    score, counts = masked_ncc(
        search, search_mask, template, template_mask
    )
    if not np.isfinite(score).any():
        raise ValueError("窗口没有足够有效覆盖，停止。")

    py, px = np.unravel_index(np.argmax(score), score.shape)
    dx, dy = int(px - radius), int(py - radius)

    alternatives = score.copy()
    alternatives[
        max(0, py-5):py+6,
        max(0, px-5):px+6,
    ] = -np.inf
    second = (
        float(alternatives.max())
        if np.isfinite(alternatives).any() else None
    )

    item = {
        "location": location["location"],
        "best_shift_working_xy": [dx, dy],
        "best_shift_uav_native_xy": [4*dx, 4*dy],
        "shift_length_uav_native_pixels":
            float(4*np.hypot(dx, dy)),
        "gradient_ncc": float(score[py, px]),
        "zero_shift_gradient_ncc":
            float(score[radius, radius]),
        "alternative_peak_outside_5_pixels": second,
        "peak_at_search_boundary":
            bool(abs(dx) == radius or abs(dy) == radius),
        "overlap_working_pixels": int(round(counts[py, px])),
    }
    results.append(item)
    print(json.dumps(item, ensure_ascii=False, indent=2), flush=True)

payload = {
    "component": "154",
    "status": "local_registration_diagnostic_only",
    "working_pixel_in_uav_native_pixels": 4,
    "search_radius_working_pixels": radius,
    "limitations": [
        "Local gradient correlation is not independent manual accuracy validation.",
        "Observed shifts are not a guaranteed error bound for all tiles.",
        "No labels or model predictions were used.",
        "No approval fields were modified.",
    ],
    "locations": results,
}

with out.open("x", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
    f.write("\n")

print("\n报告：", out)
