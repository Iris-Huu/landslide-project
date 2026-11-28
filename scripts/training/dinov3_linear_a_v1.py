"""Frozen DINOv3 linear-head A484 training; no test data."""
import argparse
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.training.exploratory import evaluate, save_checkpoint, load_checkpoint
from src.training.segmentation_steps import train_step


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


class LinearHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Conv2d(1024, 1, 1)

    def forward(self, features):
        return F.interpolate(
            self.head(features), size=(512, 512),
            mode="bilinear", align_corners=False,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("lvd1689m", "sat493m"), required=True)
    parser.add_argument("--seed", type=int, choices=(17, 42, 123), required=True)
    args = parser.parse_args()
    assert torch.cuda.is_available(), "CUDA unavailable"
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")

    cache_path = ROOT / "reports/tables/dinov3_frozen_feature_cache_v1.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    assert cache["status"] == "completed"
    manifest = ROOT / "data/interim/splits/moxitaidi_uav_spatial_split_formal_v1.csv"
    assert sha(manifest) == cache["manifest_sha256"]

    name = f"dinov3-vitl16-pretrain-{args.variant}"
    item = cache["models"][name]
    assert item["status"] == "completed"
    assert item["feature_shape"] == [1024, 32, 32]
    for filename, digest in cache["source_sha256"].items():
        assert sha(ROOT / filename) == digest, f"缓存来源代码变化：{filename}"

    sources = [
        Path(__file__),
        ROOT / "src/training/exploratory.py",
        ROOT / "src/training/segmentation_steps.py",
        ROOT / "src/metrics/binary_segmentation.py",
    ]
    plan = {
        "scope": "A484 frozen pretrained DINOv3 with linear segmentation head",
        "variants": ["lvd1689m", "sat493m"],
        "seeds": [17, 42, 123],
        "encoder": "frozen, eval, cached float32 features",
        "decoder": "Conv2d(1024,1,1), bilinear to 512x512, align_corners=False",
        "trainable_parameters": 1025,
        "decoder_initialization": "PyTorch default, seed reset before construction",
        "optimizer": "Adam",
        "learning_rate": 0.0001,
        "batch_size": 4,
        "loss": "BCE + batch-pooled soft Dice",
        "dice_smooth": 1.0,
        "bce_weight": 1.0,
        "dice_weight": 1.0,
        "max_epochs": 100,
        "minimum_epochs_before_early_stop": 30,
        "patience": 15,
        "selection": "highest validation tile-pooled IoU; ties keep earliest",
        "threshold": 0.5,
        "augmentation": "none",
        "manifest_sha256": cache["manifest_sha256"],
        "feature_cache_index_sha256": sha(cache_path),
        "valid_policy": "edge_connected_black_4_neighbors_keep_positive_v1",
        "validation_aggregation": "tile-pooled; overlaps not deduplicated",
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sources},
        "constraints": [
            "No test data used.",
            "Not full encoder fine-tuning.",
            "Not an initialization-only comparison against fully trained random baselines.",
            "A484 is not an equal-sample factorial result.",
        ],
    }
    plan_path = ROOT / "reports/tables/dinov3_linear_A_v1_training_protocol.json"
    if plan_path.exists():
        assert json.loads(plan_path.read_text(encoding="utf-8")) == plan, \
            "已固定方案与当前代码或缓存不同，停止"
    else:
        with plan_path.open("x", encoding="utf-8") as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)
            f.write("\n")

    prefix = f"uav_A_dinov3_linear_{args.variant}_seed{args.seed}_"
    assert not list((ROOT / "experiments").glob(prefix+"*")), \
        "该版本、种子已有运行目录，停止避免重复"

    # Synthetic update and head-only checkpoint restoration.
    probe = LinearHead().to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=0.0001)
    x = torch.randn((2, 1024, 32, 32), device=device)
    y = torch.zeros((2, 1, 512, 512), device=device)
    y[:, :, 128:384, 128:384] = 1
    v = torch.ones_like(y, dtype=torch.bool)
    before = probe.head.weight.detach().clone()
    train_step(probe, opt, x, y, v, loss_kind="dice_bce")
    assert not torch.equal(before, probe.head.weight)
    probe.eval()
    with torch.no_grad():
        expected = probe(x)
    with tempfile.TemporaryDirectory(dir=ROOT / "experiments") as temporary:
        path = Path(temporary) / "head.pt"
        save_checkpoint(path, probe, opt, 1, {"synthetic": True}, 0.0)
        restored = LinearHead().to(device)
        restored_opt = torch.optim.Adam(restored.parameters(), lr=0.0001)
        load_checkpoint(path, restored, restored_opt, device)
        restored.eval()
        with torch.no_grad():
            assert torch.equal(expected, restored(x))
    del probe, opt, x, y, v, before, expected, restored, restored_opt
    torch.cuda.empty_cache()
    print("合成更新与线性头检查点恢复通过。", flush=True)

    datasets = {}
    filenames = {}
    for split, expected_count in (("train", 484), ("val", 71)):
        rows = sorted(
            (r for r in item["rows"] if r["split"] == split),
            key=lambda r: r["filename"],
        )
        assert len(rows) == expected_count
        assert len({r["filename"] for r in rows}) == expected_count
        features, labels, validity = [], [], []
        for number, row in enumerate(rows, 1):
            feature_path = ROOT / row["feature"]
            target_path = ROOT / row["target"]
            assert sha(feature_path) == row["feature_sha256"], "特征哈希不符"
            assert sha(target_path) == row["target_sha256"], "标签缓存哈希不符"
            feature = np.load(feature_path, allow_pickle=False)
            assert feature.shape == (1024, 32, 32)
            assert feature.dtype == np.float32 and np.isfinite(feature).all()
            with np.load(target_path, allow_pickle=False) as target:
                label = target["label"].copy()
                valid = target["valid"].copy()
            assert label.shape == valid.shape == (512, 512)
            assert valid.dtype == np.bool_
            assert np.isin(label, (0, 1)).all()
            assert not ((~valid) & (label == 1)).any()
            features.append(feature)
            labels.append(label)
            validity.append(valid)
            if number % 100 == 0 or number == len(rows):
                print(f"载入{split}缓存：{number}/{len(rows)}", flush=True)

        datasets[split] = TensorDataset(
            torch.from_numpy(np.stack(features)).to(device),
            torch.from_numpy(np.stack(labels)).unsqueeze(1).to(
                device=device, dtype=torch.float32
            ),
            torch.from_numpy(np.stack(validity)).unsqueeze(1).to(device),
        )
        filenames[split] = [r["filename"] for r in rows]
        del features, labels, validity

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = LinearHead().to(device)
    assert sum(p.numel() for p in model.parameters()) == 1025
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    validation = DataLoader(datasets["val"], batch_size=4, shuffle=False)
    config = {
        **plan, "variant": args.variant, "seed": args.seed,
        "weight_sha256": item["weight_sha256"],
        "encoder_revision": item["revision"],
        "checkpoint_contents": "linear head and optimizer/RNG only; no encoder weights",
    }
    run = Path(tempfile.mkdtemp(prefix=prefix, dir=ROOT / "experiments"))
    record = {
        "status": "started",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "training_protocol_sha256": sha(plan_path),
        "train_samples": filenames["train"],
        "val_samples": filenames["val"],
        "environment": {"python": sys.version, "torch": torch.__version__,
                        "gpu": torch.cuda.get_device_name(0)},
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "epochs": [],
    }

    def save():
        temporary = run / "run.json.tmp"
        temporary.write_text(
            json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)+"\n",
            encoding="utf-8",
        )
        temporary.replace(run / "run.json")

    save()
    print("运行目录：", run, flush=True)
    best, stale = -1.0, 0
    started = time.perf_counter()
    try:
        for epoch in range(1, 101):
            begin = time.perf_counter()
            loader = DataLoader(
                datasets["train"], batch_size=4, shuffle=True,
                generator=torch.Generator().manual_seed(args.seed+epoch-1),
                num_workers=0,
            )
            loss_sum, valid_sum = 0.0, 0
            for x, y, valid in loader:
                n = int(valid.sum().item())
                loss = train_step(
                    model, optimizer, x, y, valid,
                    loss_kind="dice_bce",
                )
                loss_sum += loss*n
                valid_sum += n
            result = evaluate(
                model, validation, device, 0.5, loss_kind="dice_bce"
            )
            score = result["confusion"]["metrics"]["iou"]
            assert score is not None
            row = {
                "epoch": epoch, "train_loss": loss_sum/valid_sum,
                "validation": result, "seconds": time.perf_counter()-begin,
            }
            record["epochs"].append(row)
            if score > best:
                best, stale = score, 0
                record["best_epoch"] = epoch
                record["best_validation_iou"] = best
                save_checkpoint(run / "best.pt", model, optimizer, epoch, config, best)
            else:
                stale += 1
            save_checkpoint(run / "last.pt", model, optimizer, epoch, config, best)
            record["elapsed_seconds"] = time.perf_counter()-started
            save()
            print(
                f"epoch={epoch} train_loss={row['train_loss']:.4f} "
                f"val_iou={score:.4f} best={best:.4f} "
                f"seconds={row['seconds']:.1f}", flush=True,
            )
            if epoch >= 30 and stale >= 15:
                record["stop_reason"] = "validation IoU patience reached"
                break
        else:
            record["stop_reason"] = "maximum epochs reached"
        record["status"] = "completed"
    except BaseException as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()

    print("\n训练完成：", run, flush=True)
    print("最佳验证IoU：", record["best_validation_iou"], flush=True)
    print("最佳轮次：", record["best_epoch"], flush=True)


if __name__ == "__main__":
    main()
