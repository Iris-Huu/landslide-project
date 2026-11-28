import hashlib
import json
import random
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.uav_formal_v1 import FormalUAVDataset
from src.models.deeplabv3plus import build_deeplabv3plus
from src.models.segformer import build_segformer
from src.training.exploratory import evaluate, save_checkpoint
from src.training.segmentation_steps import train_step

ARCHITECTURE = "segformer"
SEED = 17
MAX_EPOCHS = 100
MIN_EPOCHS = 30
PATIENCE = 15
BATCH = 4
LR = 0.0001

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def git(*args):
    return subprocess.check_output(
        ["git", "--no-optional-locks", *args],
        cwd=ROOT, text=True,
    ).strip()

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda:0")
    manifest = ROOT / "data/interim/splits/moxitaidi_uav_spatial_split_formal_v1.csv"
    data = ROOT / "data/raw/CAS/Moxitaidi_UAV_0.6m"

    train = FormalUAVDataset(manifest, data, "train")
    val = FormalUAVDataset(manifest, data, "val")

    run = Path(tempfile.mkdtemp(
        prefix=f"uav_A_{ARCHITECTURE}_v1_seed{SEED}_",
        dir=ROOT / "experiments",
    ))

    config = {
        "architecture": ARCHITECTURE,
        "model": "SMP Segformer" if ARCHITECTURE == "segformer" else "DeepLabV3Plus",
        "encoder": "mit_b2" if ARCHITECTURE == "segformer" else "resnet50",
        "encoder_weights": None,
        "initialization": "random",
        "seed": SEED,
        "batch_size": BATCH,
        "lr": LR,
        "optimizer": "Adam",
        "max_epochs": MAX_EPOCHS,
        "minimum_epochs_before_early_stop": MIN_EPOCHS,
        "patience": PATIENCE,
        "selection": "highest validation tile-pooled IoU; ties keep earliest",
        "loss": "BCE + batch-pooled soft Dice",
        "dice_smooth": 1.0,
        "bce_weight": 1.0,
        "dice_weight": 1.0,
        "threshold": 0.5,
        "input": "512x512 full tile",
        "normalization": "RGB uint8 / 255",
        "augmentation": "none",
        "valid_policy": "edge_connected_black_4_neighbors_keep_positive_v1",
        "manifest_sha256": train.manifest_sha256,
        "train_count": len(train),
        "val_count": len(val),
        "train_order": "sorted filenames then epoch-specific seeded shuffle",
        "validation_aggregation": "tile-pooled pixels; overlaps not deduplicated",
        "test_evaluation": "not performed; deduplicated mosaic evaluation pending",
        "training_pool": "all 484 A training tiles; equal-size factorial subset not yet selected",
    }

    sources = [
        Path(__file__),
        ROOT / "src/data/uav_formal_v1.py",
        ROOT / "src/data/uav_data.py",
        ROOT / "src/data/nodata_v1.py",
        ROOT / "src/models/deeplabv3plus.py",
        ROOT / "src/models/segformer.py",
        ROOT / "src/training/segmentation_steps.py",
        ROOT / "src/training/exploratory.py",
        ROOT / "src/metrics/binary_segmentation.py",
    ]

    record = {
        "status": "started",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "git_head": git("rev-parse", "HEAD"),
        "git_status": git("status", "--short", "--untracked-files=all"),
        "source_sha256": {
            str(p.relative_to(ROOT)): sha(p) for p in sources
        },
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "train_samples": [s.filename for s in train.samples],
        "val_samples": [s.filename for s in val.samples],
        "epochs": [],
        "limitations": [
            "Random initialization; not a pretrained model comparison.",
            "No independent test results yet.",
            "NoData rule uses positive labels as specified by supervisor feedback.",
            "Cross-platform B isolation remains unresolved.",
        ],
    }

    def write_record():
        temporary = run / "run.json.tmp"
        temporary.write_text(
            json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(run / "run.json")

    write_record()
    print("运行目录：", run, flush=True)

    try:
        model, provenance = (
            build_segformer("mit_b2", None)
            if ARCHITECTURE == "segformer"
            else build_deeplabv3plus("resnet50", None)
        )
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        record["weights"] = provenance
        record["parameter_count"] = sum(p.numel() for p in model.parameters())

        validation_loader = DataLoader(
            val, batch_size=BATCH, shuffle=False, num_workers=0
        )
        best = -1.0
        stale = 0
        start_time = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)

        for epoch in range(1, MAX_EPOCHS + 1):
            begin = time.perf_counter()
            loader = DataLoader(
                train,
                batch_size=BATCH,
                shuffle=True,
                generator=torch.Generator().manual_seed(SEED + epoch - 1),
                num_workers=0,
            )
            loss_sum = 0.0
            valid_sum = 0

            for x, y, valid in loader:
                x = x.to(device)
                y = y.to(device)
                valid = valid.to(device)
                n = int(valid.sum().item())

                loss = train_step(
                    model, optimizer, x, y, valid,
                    loss_kind="dice_bce",
                    dice_smooth=1.0,
                    bce_weight=1.0,
                    dice_weight=1.0,
                )
                loss_sum += loss * n
                valid_sum += n

            validation = evaluate(
                model, validation_loader, device, 0.5,
                loss_kind="dice_bce",
                dice_smooth=1.0,
                bce_weight=1.0,
                dice_weight=1.0,
            )
            torch.cuda.synchronize()
            score = validation["confusion"]["metrics"]["iou"]
            if score is None:
                raise RuntimeError("Validation IoU undefined")

            item = {
                "epoch": epoch,
                "train_loss": loss_sum / valid_sum,
                "validation": validation,
                "seconds": time.perf_counter() - begin,
            }
            record["epochs"].append(item)

            if score > best:
                best = score
                stale = 0
                record["best_epoch"] = epoch
                record["best_validation_iou"] = best
                save_checkpoint(
                    run / "best.pt", model, optimizer, epoch, config, best
                )
            else:
                stale += 1

            save_checkpoint(
                run / "last.pt", model, optimizer, epoch, config, best
            )
            record["elapsed_seconds"] = time.perf_counter() - start_time
            record["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            write_record()

            print(
                f"epoch={epoch} train_loss={item['train_loss']:.4f} "
                f"val_loss={validation['loss']:.4f} "
                f"val_iou={score:.4f} best={best:.4f} "
                f"seconds={item['seconds']:.1f}",
                flush=True,
            )

            if epoch >= MIN_EPOCHS and stale >= PATIENCE:
                record["stop_reason"] = "validation IoU patience reached"
                break
        else:
            record["stop_reason"] = "maximum epochs reached"

        record["status"] = "completed"
    except BaseException as exc:
        record["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_record()

    print("训练结束：", run, flush=True)
    print("最佳验证IoU：", record["best_validation_iou"], flush=True)
    print("最佳轮次：", record["best_epoch"], flush=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--architecture", choices=("segformer", "deeplabv3plus"), default="segformer")
    args = parser.parse_args()
    SEED = args.seed
    ARCHITECTURE = args.architecture
    main()
