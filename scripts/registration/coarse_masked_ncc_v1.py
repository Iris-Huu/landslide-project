import json
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import correlate

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "data/interim/registration/mosaics_v1"
OUT = ROOT / "data/interim/registration/coarse_masked_ncc_v1.json"
EXTRA_DOWNSAMPLE = 4

cv2.setNumThreads(2)

def corr(a, b):
    return correlate(a, b, mode="full", method="fft").astype(np.float32)

def masked_ncc(a, ma, b, mb, minimum_overlap):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    ma = ma.astype(np.float32)
    mb = mb.astype(np.float32)

    n = np.maximum(np.rint(corr(ma, mb)), 0)
    sa = corr(a * ma, mb)
    sb = corr(ma, b * mb)
    saa = corr(a * a * ma, mb)
    sbb = corr(ma, b * b * mb)
    sab = corr(a * ma, b * mb)

    safe_n = np.maximum(n, 1)
    covariance = sab - sa * sb / safe_n
    va = np.maximum(saa - sa * sa / safe_n, 0)
    vb = np.maximum(sbb - sb * sb / safe_n, 0)
    denominator = np.sqrt(va * vb)

    score = np.full(n.shape, -np.inf, dtype=np.float32)
    eligible = (n >= minimum_overlap) & (denominator > 1e-6)
    score[eligible] = np.clip(
        covariance[eligible] / denominator[eligible], -1, 1
    )
    return score, n

# 检查相关峰对应的平移方向。
rng = np.random.default_rng(42)
small = rng.normal(size=(12, 10)).astype(np.float32)
canvas = np.zeros((30, 30), dtype=np.float32)
canvas[7:19, 9:19] = small
test_score, _ = masked_ncc(
    canvas, np.ones_like(canvas),
    small, np.ones_like(small),
    small.size,
)
y, x = np.unravel_index(np.argmax(test_score), test_score.shape)
assert (x - small.shape[1] + 1, y - small.shape[0] + 1) == (9, 7)
print("合成平移方向检查：通过", flush=True)

if OUT.exists():
    raise SystemExit(f"输出已存在，未覆盖：{OUT}")

def load(path):
    with np.load(path, allow_pickle=False) as data:
        gray = data["gray"]
        coverage = data["coverage"]
        metadata = json.loads(str(data["metadata"].item()))

    h, w = gray.shape
    size = (w // EXTRA_DOWNSAMPLE, h // EXTRA_DOWNSAMPLE)
    gray = cv2.resize(
        gray, size, interpolation=cv2.INTER_AREA
    ).astype(np.float32) / 255.0
    coverage_fraction = cv2.resize(
        coverage.astype(np.float32), size, interpolation=cv2.INTER_AREA
    )
    mask = (coverage_fraction > 0.999).astype(np.uint8)
    mask = cv2.erode(
        mask, np.ones((5, 5), np.uint8),
        borderType=cv2.BORDER_CONSTANT, borderValue=0,
    )
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gx * gx + gy * gy)
    if mask.any():
        upper = float(np.quantile(gradient[mask > 0], 0.99))
        if upper > 0:
            gradient = np.minimum(gradient, upper) / upper
    return gradient, mask, metadata

uav, uav_mask, uav_meta = load(DIR / "uav_component_2.npz")
results = []

for component in ("154", "158", "148"):
    print(f"\nSAT 连通组 {component}：计算梯度互相关……", flush=True)
    sat, sat_mask, sat_meta = load(
        DIR / f"satellite_component_{component}.npz"
    )

    minimum_overlap = max(
        2000,
        int(0.15 * min(int(uav_mask.sum()), int(sat_mask.sum()))),
    )
    score, overlap = masked_ncc(
        uav, uav_mask, sat, sat_mask, minimum_overlap
    )

    peaks = []
    for rank in range(1, 6):
        y, x = np.unravel_index(np.argmax(score), score.shape)
        value = float(score[y, x])
        if not np.isfinite(value):
            break

        dx = int(x - sat.shape[1] + 1)
        dy = int(y - sat.shape[0] + 1)
        peak = {
            "rank": rank,
            "ncc": value,
            "overlap_coarse_pixels": int(overlap[y, x]),
            "translation_coarse_xy": [dx, dy],
            "translation_mosaic_working_xy": [
                dx * EXTRA_DOWNSAMPLE,
                dy * EXTRA_DOWNSAMPLE,
            ],
        }
        peaks.append(peak)
        print(peak, flush=True)

        radius = 24
        score[
            max(0, y-radius):min(score.shape[0], y+radius+1),
            max(0, x-radius):min(score.shape[1], x+radius+1),
        ] = -np.inf

    results.append({
        "sat_component": component,
        "status": "coarse_candidates_only",
        "minimum_overlap_coarse_pixels": minimum_overlap,
        "sat_metadata": sat_meta,
        "peaks": peaks,
    })
    del score, overlap

report = {
    "method": "gradient magnitude masked normalized cross-correlation",
    "extra_downsample": EXTRA_DOWNSAMPLE,
    "coarse_pixel_in_uav_native_pixels": 16,
    "transform": "uav coarse xy = satellite coarse xy + translation",
    "uav_metadata": uav_meta,
    "limitations": [
        "尺度2、旋转0为固定假设。",
        "粗定位候选，尚未进行高分辨率细化及独立验证。",
        "低重叠或重复纹理可能产生假峰。",
        "本结果不能直接作为B单元可用图块批准依据。",
    ],
    "results": results,
}

with OUT.open("x", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
    f.write("\n")

print("\n保存：", OUT)
