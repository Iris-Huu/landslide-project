"""Evaluate fixed A checkpoints once under frozen test protocol v2."""
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "2"

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import rasterio
import scipy
import segmentation_models_pytorch as smp
import torch
from rasterio.enums import ColorInterp
from rasterio.errors import NotGeoreferencedWarning

from src.data.nodata_v1 import valid_mask_v1
from src.metrics.boundary_v1 import boundary_counts, boundary_metrics
from src.metrics.mosaic_v1 import ProbabilityMosaic
from src.metrics.spatial_bootstrap_v1 import METRICS, block_counts, bootstrap, metrics
from src.models.deeplabv3plus import build_deeplabv3plus

PROTOCOL = ROOT / "reports/tables/uav_A_v1_test_protocol_v2.json"
EXPECTED_PROTOCOL = "c2a91f680f4e91eedfadbbca24950117043f82111d728fcfaa87ea379d6bcccc"
DATA = ROOT / "data/raw/CAS/Moxitaidi_UAV_0.6m"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure(condition, message):
    if not condition:
        raise ValueError(message)


def read_tile(name):
    ensure(
        name and Path(name).name == name and "/" not in name and "\\" not in name,
        "Unsafe filename",
    )
    arrays, hashes = {}, {}
    for folder, count in (("img", 3), ("mask", 1)):
        path = DATA / folder / name
        hashes[folder] = sha(path)
        with warnings.catch_warnings(), rasterio.Env(GDAL_PAM_ENABLED="NO"):
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with rasterio.open(path) as ds:
                ensure(
                    ds.count == count and ds.width == 512 and ds.height == 512,
                    f"{name}: invalid dimensions or bands",
                )
                ensure(all(d == "uint8" for d in ds.dtypes), f"{name}: invalid dtype")
                if folder == "img":
                    ensure(
                        ds.colorinterp == (
                            ColorInterp.red, ColorInterp.green, ColorInterp.blue
                        ),
                        f"{name}: RGB interpretation mismatch",
                    )
                arrays[folder] = ds.read()
    raw, label = arrays["img"], arrays["mask"][0]
    ensure(np.isin(label, (0, 1)).all(), f"{name}: invalid labels")
    valid = valid_mask_v1(raw, label)
    ensure(valid.dtype == np.bool_, "Invalid validity type")
    ensure(not ((~valid) & (label == 1)).any(), "Positive labels ignored")
    return raw, label, valid, hashes


def clean_metrics(values):
    return {
        name: float(value) if np.isfinite(value) else None
        for name, value in zip(METRICS, values)
    }


def main():
    ensure(sha(PROTOCOL) == EXPECTED_PROTOCOL, "Protocol v2 hash mismatch")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    ensure(protocol["spatial_bootstrap"]["replicates"] == 10000, "Wrong bootstrap count")
    ensure(protocol["inference"]["threshold"] == 0.5, "Wrong threshold")
    ensure(protocol["boundary_evaluation"]["tolerance_pixels"] == 2, "Wrong boundary tolerance")
    ensure(protocol["spatial_bootstrap"]["block_side_pixels"] == 833, "Wrong block size")

    for name, expected in protocol["source_sha256"].items():
        ensure(sha(ROOT / name) == expected, f"Source changed: {name}")

    manifest = ROOT / protocol["manifest"]
    ensure(sha(manifest) == protocol["manifest_sha256"], "Manifest changed")
    rows = list(csv.DictReader(io.StringIO(
        manifest.read_bytes().decode("utf-8-sig")
    )))
    ensure(
        Counter(r["split_v1"] for r in rows) == Counter({
            "train": 484, "val": 71, "test": 149, "buffer": 57, "excluded": 223
        }),
        "Frozen split counts mismatch",
    )
    rows = sorted(
        (r for r in rows if r["split_v1"] == "test"),
        key=lambda r: r["filename"],
    )
    ensure(len(rows) == 149, "Wrong test count")
    ensure(len({r["filename"] for r in rows}) == 149, "Duplicate test names")
    ensure(
        all(r["platform"] == "UAV" and r["component"] == "2" for r in rows),
        "Unexpected test platform or component",
    )

    for item in protocol["runs"]:
        for path_key, hash_key in (
            ("run_json", "run_json_sha256"),
            ("checkpoint", "checkpoint_sha256"),
        ):
            ensure(
                sha(ROOT / item[path_key]) == item[hash_key],
                f"Input hash mismatch: {item[path_key]}",
            )

    ensure(torch.cuda.is_available(), "CUDA unavailable")
    for old in (ROOT / "experiments").glob("uav_A_v1_test_*/run.json"):
        record = json.loads(old.read_text(encoding="utf-8"))
        ensure(
            record.get("status") != "completed",
            f"A completed evaluation already exists: {old}; do not repeat",
        )

    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")

    run = Path(tempfile.mkdtemp(
        prefix="uav_A_v1_test_", dir=ROOT / "experiments"
    ))
    record = {
        "status": "started",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": sha(PROTOCOL),
        "evaluation_source_sha256": sha(Path(__file__)),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True
        ).strip(),
        "environment": {
            "python": sys.version, "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "numpy": np.__version__, "scipy": scipy.__version__,
            "rasterio": rasterio.__version__, "smp": smp.__version__,
            "gpu": torch.cuda.get_device_name(0),
        },
        "input_files": [],
        "results": [],
        "limitations": protocol["constraints"] + [
            "833 pixels corresponds to about 500 m only under nominal 0.6 m spacing.",
            "Boundary F1 excludes the protocol-defined coverage-edge band.",
            "Bootstrap intervals are conditional on this test region and fixed models.",
            "A484 is not an equal-sample factorial comparison.",
        ],
    }

    def save():
        temporary = run / "run.json.tmp"
        temporary.write_text(
            json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(run / "run.json")

    save()
    print("评价目录：", run, flush=True)
    try:
        top = min(int(r["grid_row"]) for r in rows) * 256
        left = min(int(r["grid_col"]) for r in rows) * 256
        bottom = max(int(r["grid_row"]) for r in rows) * 256 + 512
        right = max(int(r["grid_col"]) for r in rows) * 256 + 512
        ensure(top >= 0 and left >= 0, "Negative component coordinates")
        shape = (bottom-top, right-left)
        record["mosaic"] = {"origin_native_xy": [left, top], "shape_hw": list(shape)}

        # Audit label consistency before model inference.
        geometry = ProbabilityMosaic(*shape)
        tiles = []
        for number, row in enumerate(rows, 1):
            name = row["filename"]
            raw, label, valid, hashes = read_tile(name)
            y = int(row["grid_row"]) * 256 - top
            x = int(row["grid_col"]) * 256 - left
            geometry.add(np.zeros(label.shape), label, valid, y, x)
            tiles.append((raw, label, valid, y, x))
            record["input_files"].append({"filename": name, "sha256": hashes})
            if number % 25 == 0 or number == len(rows):
                print(f"读取与重叠标签核验：{number}/{len(rows)}", flush=True)

        _, reference_truth, reference_valid = geometry.result()
        record["mosaic"].update({
            "unique_valid_pixels": int(reference_valid.sum()),
            "valid_tile_pixel_contributions": int(geometry.count.sum(dtype=np.uint64)),
            "pixels_with_multiple_valid_contributions": int((geometry.count > 1).sum()),
        })
        del geometry
        save()

        all_counts, block_ids = [], None
        for item in protocol["runs"]:
            model, _ = build_deeplabv3plus("resnet50", None)
            # Trusted, hash-verified local checkpoint; contains optimizer/RNG state.
            checkpoint = torch.load(
                ROOT / item["checkpoint"], map_location="cpu", weights_only=False
            )
            training = json.loads(
                (ROOT / item["run_json"]).read_text(encoding="utf-8")
            )
            ensure(checkpoint["config"] == training["config"], "Checkpoint config mismatch")
            ensure(checkpoint["config"]["seed"] == item["seed"], "Checkpoint seed mismatch")
            ensure(checkpoint["epoch"] == item["best_epoch"], "Checkpoint epoch mismatch")
            model.load_state_dict(checkpoint["model"], strict=True)
            del checkpoint
            model = model.to(device).eval()
            mosaic = ProbabilityMosaic(*shape)

            with torch.inference_mode():
                # Synthetic shape/finite check before processing real tiles.
                check = model(torch.zeros((1, 3, 512, 512), device=device))
                ensure(
                    tuple(check.shape) == (1, 1, 512, 512)
                    and bool(torch.isfinite(check).all()),
                    "Synthetic model output check failed",
                )
                del check
                for start in range(0, len(tiles), 4):
                    batch = tiles[start:start+4]
                    images = torch.from_numpy(
                        np.stack([tile[0] for tile in batch])
                    ).to(device=device, dtype=torch.float32) / 255.0
                    logits = model(images)
                    ensure(
                        tuple(logits.shape) == (len(batch), 1, 512, 512)
                        and bool(torch.isfinite(logits).all()),
                        "Invalid test logits",
                    )
                    probability = torch.sigmoid(logits).cpu().numpy()[:, 0]
                    for p, (_, label, valid, y, x) in zip(probability, batch):
                        mosaic.add(p, label, valid, y, x)
                    del images, logits
                    if start % 40 == 0:
                        print(
                            f"seed {item['seed']}：{min(start+4, len(tiles))}/{len(tiles)}",
                            flush=True,
                        )

            probability, truth, valid = mosaic.result()
            ensure(np.array_equal(valid, reference_valid), "Validity differs between seeds")
            ensure(np.array_equal(truth[valid], reference_truth[valid]), "Truth differs")
            prediction = probability >= 0.5
            ids, counts = block_counts(prediction, truth, valid, top, left, side=833)
            if block_ids is None:
                block_ids = ids
            else:
                ensure(np.array_equal(ids, block_ids), "Spatial blocks differ")

            total = counts.sum(axis=0)
            ensure(int(total.sum()) == int(valid.sum()), "Pixel conservation failed")
            boundary = boundary_counts(prediction, truth, valid, tolerance=2)
            result = {
                "seed": item["seed"],
                "checkpoint": item["checkpoint"],
                "checkpoint_sha256": item["checkpoint_sha256"],
                "selected_epoch": item["best_epoch"],
                "confusion": dict(zip(
                    ("tp", "fp", "fn", "tn"), map(int, total)
                )),
                "metrics": clean_metrics(metrics(total)),
                "boundary_counts": boundary,
                "boundary_metrics": boundary_metrics(boundary),
                "spatial_blocks": int(len(ids)),
            }
            record["results"].append(result)
            all_counts.append(counts)
            save()
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
            del model, mosaic, probability, prediction, truth, valid
            torch.cuda.empty_cache()

        stacked = np.stack(all_counts)
        np.savez_compressed(
            run / "spatial_block_counts.npz",
            block_ids=block_ids, counts=stacked,
            seeds=np.array([r["seed"] for r in protocol["runs"]]),
        )
        record["spatial_block_counts_sha256"] = sha(run / "spatial_block_counts.npz")
        print("计算10,000次配对空间bootstrap……", flush=True)
        record["spatial_bootstrap"] = bootstrap(
            stacked, replicates=10000, seed=20261127
        )
        summary = {}
        for name in (*METRICS, "boundary_f1"):
            values = [
                r["boundary_metrics"][name] if name == "boundary_f1"
                else r["metrics"][name]
                for r in record["results"]
            ]
            complete = all(value is not None for value in values)
            summary[name] = {
                "seed_mean": float(np.mean(values)) if complete else None,
                "seed_sample_sd": float(np.std(values, ddof=1)) if complete else None,
            }
        record["seed_summary"] = summary
        record["status"] = "completed"
        print("\n三种子汇总：", json.dumps(summary, ensure_ascii=False, indent=2))
    except BaseException as exc:
        record["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
    print("\n评价完成：", run / "run.json", flush=True)


if __name__ == "__main__":
    main()
