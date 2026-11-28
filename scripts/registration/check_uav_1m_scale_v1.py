"""Geometry-only scale checks against UAV training-area imagery."""
import csv
import hashlib
import json
import warnings
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/tables/uav_1m_relative_scale_v1.json"
MANIFEST = ROOT / "data/interim/splits/moxitaidi_uav_spatial_split_formal_v1.csv"
MOSAIC = ROOT / "data/interim/registration/mosaics_v1/uav_component_2.npz"
SOURCE = ROOT / "data/raw/CAS/Moxitaidi_UAV_1m/img"

if OUT.exists():
    raise SystemExit("尺度报告已存在，未覆盖。")

cv2.setNumThreads(2)
cv2.setRNGSeed(42)

metadata = json.loads(MANIFEST.with_suffix(".json").read_text(encoding="utf-8"))
manifest_hash = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
assert manifest_hash == metadata["formal_manifest_sha256"]

with MANIFEST.open(encoding="utf-8-sig", newline="") as f:
    train = [r for r in csv.DictReader(f) if r["split_v1"] == "train"]
assert len(train) == 484

with np.load(MOSAIC, allow_pickle=False) as data:
    reference = data["gray"].copy()
    coverage = data["coverage"].astype(bool)
    meta = json.loads(str(data["metadata"].item()))
assert meta["downsample_factor"] == 4
assert meta["native_grid_step"] == 256
assert meta["native_tile_size"] == 512

# Feature locations are restricted to known training-tile footprints.
mask = np.zeros(reference.shape, dtype=np.uint8)
for row in train:
    y = (int(row["grid_row"])-meta["grid_origin_row"]) * 64
    x = (int(row["grid_col"])-meta["grid_origin_col"]) * 64
    assert y >= 0 and x >= 0
    mask[y:y+128, x:x+128] = 1
mask &= coverage.astype(np.uint8)
mask = cv2.erode(mask, np.ones((13, 13), np.uint8)) * 255

sift = cv2.SIFT_create(nfeatures=40000, contrastThreshold=0.02)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
rk, rd = sift.detectAndCompute(clahe.apply(reference), mask)
if rd is None:
    raise SystemExit("训练覆盖区域未提取到特征。")

matcher = cv2.FlannBasedMatcher(
    dict(algorithm=1, trees=5), dict(checks=128)
)
matcher.add([rd])
matcher.train()

files = sorted(
    p for p in SOURCE.iterdir()
    if p.is_file() and p.suffix.lower() in (".tif", ".tiff")
)
assert files, "1m影像目录为空"
indices = np.unique(np.linspace(0, len(files)-1, min(80, len(files)), dtype=int))
selected = [files[int(i)] for i in indices]

report = {
    "status": "started",
    "scope": "Relative scale diagnostic; no labels or model predictions",
    "manifest_sha256": manifest_hash,
    "reference_mosaic_sha256": hashlib.sha256(MOSAIC.read_bytes()).hexdigest(),
    "source_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "reference_feature_count": len(rk),
    "reference_downsample_factor": 4,
    "source_downsample_factor": 2,
    "selection": "80 uniformly spaced indices of sorted image filenames, or all if fewer",
    "selected_files": [p.name for p in selected],
    "support_rule": {
        "minimum_unique_matches": 15,
        "minimum_fit_inliers": 8,
        "minimum_holdout_support_within_3_working_pixels": 4,
        "minimum_fit_inlier_span_source_working_xy": [40, 40],
        "scale_acceptance_filter": "none; nominal ratio is not used to select matches",
    },
    "results": [],
    "limitations": [
        "Accepted matches are geometric diagnostics, not independent manual accuracy measurements.",
        "A relative ratio agreeing with nominal values cannot independently prove absolute ground resolution.",
        "Reference features are restricted to frozen training-tile coverage.",
    ],
}

def save():
    temporary = OUT.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(OUT)

OUT.parent.mkdir(parents=True, exist_ok=True)
save()
print("训练区域特征数：", len(rk), "；固定检查影像数：", len(selected), flush=True)

for number, path in enumerate(selected, 1):
    item = {"filename": path.name, "supported": False}
    with warnings.catch_warnings(), rasterio.Env(GDAL_PAM_ENABLED="NO"):
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(path) as ds:
            assert ds.count == 3 and ds.width == ds.height == 512
            assert all(d == "uint8" for d in ds.dtypes)
            rgb = np.moveaxis(ds.read(), 0, -1)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    sk, sd = sift.detectAndCompute(clahe.apply(gray), None)
    pairs, used_source, used_target = [], set(), set()
    if sd is not None:
        candidates = []
        for pair in matcher.knnMatch(sd, k=2):
            if len(pair) == 2 and pair[0].distance < 0.70 * pair[1].distance:
                candidates.append(pair[0])
        for match in sorted(candidates, key=lambda m: m.distance):
            a = tuple(round(float(v), 1) for v in sk[match.queryIdx].pt)
            b = tuple(round(float(v), 1) for v in rk[match.trainIdx].pt)
            if a in used_source or b in used_target:
                continue
            used_source.add(a)
            used_target.add(b)
            pairs.append((a, b))

    item["unique_matches"] = len(pairs)
    if len(pairs) >= 15:
        source = np.float32([a for a, b in pairs])
        target = np.float32([b for a, b in pairs])
        order = np.random.default_rng(42).permutation(len(pairs))
        holdout = order[::3]
        fit = np.setdiff1d(order, holdout)
        cv2.setRNGSeed(42)
        matrix, flags = cv2.estimateAffinePartial2D(
            source[fit], target[fit], method=cv2.RANSAC,
            ransacReprojThreshold=3.0, maxIters=10000,
            confidence=0.999, refineIters=10,
        )
        if matrix is not None:
            inside = flags.ravel().astype(bool)
            span = np.ptp(source[fit][inside], axis=0) if inside.any() else np.zeros(2)
            errors = np.linalg.norm(
                source @ matrix[:, :2].T + matrix[:, 2] - target, axis=1
            )
            supported = int((errors[holdout] <= 3).sum())
            scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
            item.update({
                "fit_inliers": int(inside.sum()),
                "fit_span_source_working_xy": span.tolist(),
                "holdout_matches": len(holdout),
                "holdout_supported": supported,
                "holdout_median_error_working_pixels": float(np.median(errors[holdout])),
                "working_scale": scale,
                "native_1m_to_06m_pixel_scale": scale * 4 / 2,
                "rotation_degrees": float(np.degrees(
                    np.arctan2(matrix[1, 0], matrix[0, 0])
                )),
                "matrix_source_working_to_reference_working": matrix.tolist(),
                "supported": bool(
                    inside.sum() >= 8 and supported >= 4 and (span >= 40).all()
                ),
            })

    report["results"].append(item)
    save()
    if item["supported"]:
        print(
            f"{path.name}：尺度={item['native_1m_to_06m_pixel_scale']:.5f}，"
            f"旋转={item['rotation_degrees']:.3f}°，"
            f"拟合内点={item['fit_inliers']}，留出支持={item['holdout_supported']}",
            flush=True,
        )
    elif number % 10 == 0:
        print(f"已检查：{number}/{len(selected)}", flush=True)

values = [
    r["native_1m_to_06m_pixel_scale"]
    for r in report["results"] if r["supported"]
]
report["summary"] = {
    "checked": len(selected),
    "supported": len(values),
    "nominal_ratio_for_comparison_only": 1 / 0.6,
    "median_native_scale": float(np.median(values)) if values else None,
    "minimum_native_scale": float(min(values)) if values else None,
    "maximum_native_scale": float(max(values)) if values else None,
}
report["status"] = "completed"
save()
print("\n汇总：", json.dumps(report["summary"], ensure_ascii=False, indent=2))
print("报告：", OUT)
