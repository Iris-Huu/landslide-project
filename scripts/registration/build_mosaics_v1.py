import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/interim/registration/mosaics_v1"
OUT.mkdir(parents=True, exist_ok=True)
cv2.setNumThreads(2)

# 工作尺度：SAT原图缩小2倍，UAV原图缩小4倍。
# 基于SAT像元约为UAV两倍的工作假设，两边在此尺度上比较平移。
TASKS = [
    ("uav", "2", "Moxitaidi_UAV_0.6m", 4),
    ("satellite", "154", "Moxitaidi_SAT", 2),
    ("satellite", "158", "Moxitaidi_SAT", 2),
    ("satellite", "148", "Moxitaidi_SAT", 2),
]

for platform, component, dataset, factor in TASKS:
    target = OUT / f"{platform}_component_{component}.npz"
    if target.exists():
        raise SystemExit(f"输出已存在，未覆盖：{target}")

for platform, component, dataset, factor in TASKS:
    grid = ROOT / (
        f"data/interim/reconstructed_grids/"
        f"{platform}_grid_coordinates_v2.csv"
    )
    with grid.open("r", encoding="utf-8-sig", newline="") as f:
        rows = [
            r for r in csv.DictReader(f)
            if r["component"] == component
        ]
    if not rows:
        raise RuntimeError(f"找不到连通组：{platform}/{component}")

    coords = [(int(r["grid_row"]), int(r["grid_col"])) for r in rows]
    if len(set(coords)) != len(coords):
        raise RuntimeError("存在重复网格坐标")

    min_row = min(r for r, c in coords)
    min_col = min(c for r, c in coords)
    max_row = max(r for r, c in coords)
    max_col = max(c for r, c in coords)

    step = 256 // factor
    tile = 512 // factor
    height = (max_row - min_row) * step + tile
    width = (max_col - min_col) * step + tile

    total = np.zeros((height, width), dtype=np.float32)
    count = np.zeros((height, width), dtype=np.uint16)

    print(
        f"\n开始 {platform}/{component}："
        f"{len(rows)} 张，工作拼图 {height}×{width}",
        flush=True,
    )

    for index, row in enumerate(rows, 1):
        image_path = ROOT / "data/raw/CAS" / dataset / "img" / row["filename"]
        with rasterio.open(image_path) as src:
            rgb = src.read()

        if rgb.shape != (3, 512, 512) or rgb.dtype != np.uint8:
            raise RuntimeError(f"影像格式不符：{image_path}")

        rgb = np.moveaxis(rgb, 0, -1)
        small = cv2.resize(rgb, (tile, tile), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)

        y = (int(row["grid_row"]) - min_row) * step
        x = (int(row["grid_col"]) - min_col) * step
        total[y:y + tile, x:x + tile] += gray
        count[y:y + tile, x:x + tile] += 1

        if index % 100 == 0 or index == len(rows):
            print(f"  已读取 {index}/{len(rows)}", flush=True)

    coverage = count > 0
    gray = np.zeros((height, width), dtype=np.uint8)
    gray[coverage] = np.rint(
        total[coverage] / count[coverage]
    ).astype(np.uint8)

    metadata = {
        "platform": platform,
        "component": component,
        "tiles": len(rows),
        "downsample_factor": factor,
        "grid_origin_row": min_row,
        "grid_origin_col": min_col,
        "native_grid_step": 256,
        "native_tile_size": 512,
        "working_shape": [height, width],
        "grid_sha256": hashlib.sha256(grid.read_bytes()).hexdigest(),
        "overlap_policy": "mean of available grayscale tile pixels",
        "purpose": "geometry-only registration; no labels or model predictions",
        "scale_assumption": "SAT native pixel approximately 2 UAV native pixels",
    }

    target = OUT / f"{platform}_component_{component}.npz"
    with target.open("xb") as f:
        np.savez_compressed(
            f,
            gray=gray,
            coverage=coverage,
            metadata=json.dumps(metadata, ensure_ascii=False),
        )

    preview_scale = min(1.0, 1600 / max(height, width))
    preview = cv2.resize(
        gray,
        (
            max(1, round(width * preview_scale)),
            max(1, round(height * preview_scale)),
        ),
        interpolation=cv2.INTER_AREA,
    )
    preview_path = target.with_suffix(".png")
    if not cv2.imwrite(str(preview_path), preview):
        raise RuntimeError(f"预览图写入失败：{preview_path}")

    print("保存：", target, flush=True)
    print("预览：", preview_path, flush=True)
    print("有影像覆盖的工作像素：", int(coverage.sum()), flush=True)

    del total, count, coverage, gray, preview

print("\n四张配准拼图已生成。")
