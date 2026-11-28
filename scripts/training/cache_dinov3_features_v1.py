"""Cache frozen, float32 DINOv3 training/validation features offline."""
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from src.data.uav_formal_v1 import FormalUAVDataset
from src.models.dinov3_segmentation import build_local_dinov3

OUT = ROOT / "reports/tables/dinov3_frozen_feature_cache_v1.json"
if OUT.exists():
    raise SystemExit("特征缓存索引已存在，未重复执行。")

def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()

assert torch.cuda.is_available(), "CUDA unavailable"
torch.set_num_threads(2)
torch.manual_seed(17)
torch.cuda.manual_seed_all(17)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

manifest = ROOT / "data/interim/splits/moxitaidi_uav_spatial_split_formal_v1.csv"
data_dir = ROOT / "data/raw/CAS/Moxitaidi_UAV_0.6m"
datasets = {
    split: FormalUAVDataset(manifest, data_dir, split)
    for split in ("train", "val")
}
assert len(datasets["train"]) == 484
assert len(datasets["val"]) == 71

required = 2 * 555 * 1024 * 32 * 32 * 4
free = shutil.disk_usage(ROOT).free
print("可用空间：", round(free / 1024**3, 2), "GiB", flush=True)
print("预计特征空间：", round(required / 1024**3, 2), "GiB", flush=True)
assert free > required + 2*1024**3, "磁盘余量不足，停止"

weights = ROOT / "weights/dinov3"
checks = json.loads((weights / "checksums.json").read_text(encoding="utf-8"))
versions = json.loads(
    (weights / "download_record.json").read_text(encoding="utf-8")
)
sources = [
    Path(__file__),
    ROOT / "src/models/dinov3_segmentation.py",
    ROOT / "src/data/uav_formal_v1.py",
    ROOT / "src/data/uav_data.py",
    ROOT / "src/data/nodata_v1.py",
]
record = {
    "status": "started",
    "purpose": "Frozen-encoder linear segmentation baseline",
    "manifest_sha256": sha(manifest),
    "splits": {"train": 484, "val": 71},
    "feature_dtype": "float32",
    "augmentation": "none",
    "source_sha256": {
        str(p.relative_to(ROOT)): sha(p) for p in sources
    },
    "models": {},
    "limitations": [
        "Not full encoder fine-tuning.",
        "Frozen pretrained encoder versus fully trained random baselines is not a controlled initialization-only comparison.",
        "No test images or labels are read by this script.",
    ],
}

def save():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(OUT)

save()

for name in (
    "dinov3-vitl16-pretrain-lvd1689m",
    "dinov3-vitl16-pretrain-sat493m",
):
    print("\n缓存：", name, flush=True)
    weight_file = weights / name / "model.safetensors"
    expected = checks[f"{name}/model.safetensors"]["sha256"]
    assert sha(weight_file) == expected, "权重哈希变化"

    cache = Path(tempfile.mkdtemp(
        prefix=f"dinov3_features_{name.split('-')[-1]}_",
        dir=ROOT / "experiments",
    ))
    model = build_local_dinov3(weights / name).cuda().eval()
    assert all(not p.requires_grad for p in model.encoder.parameters())
    assert model.patch_size == 16
    assert isinstance(model.head, torch.nn.Conv2d)
    assert model.head.in_channels == 1024
    assert model.head.out_channels == 1
    assert model.head.kernel_size == (1, 1)

    item = {
        "status": "started",
        "revision": versions[name]["revision"],
        "weight_sha256": expected,
        "cache_directory": str(cache.relative_to(ROOT)),
        "feature_shape": [1024, 32, 32],
        "image_mean": model.mean.detach().cpu().flatten().tolist(),
        "image_std": model.std.detach().cpu().flatten().tolist(),
        "rows": [],
    }
    record["models"][name] = item
    save()
    checked_equivalence = False

    with torch.inference_mode():
        for split, dataset in datasets.items():
            destination = cache / split
            destination.mkdir()
            for start in range(0, len(dataset), 4):
                indices = list(range(start, min(start+4, len(dataset))))
                batch = [dataset[i] for i in indices]
                images = torch.stack([row[0] for row in batch]).cuda()

                hidden = model.encoder(
                    pixel_values=(images-model.mean)/model.std
                ).last_hidden_state
                tokens = hidden[:, model.special_tokens:, :]
                assert tokens.shape[1:] == (1024, 1024)
                spatial = tokens.transpose(1, 2).reshape(
                    len(indices), 1024, 32, 32
                ).contiguous()

                if not checked_equivalence:
                    direct = model(images)
                    cached_path = F.interpolate(
                        model.head(spatial), size=(512, 512),
                        mode="bilinear", align_corners=False,
                    )
                    difference = float((direct-cached_path).abs().max().item())
                    assert torch.allclose(
                        direct, cached_path, atol=1e-5, rtol=1e-5
                    ), "缓存路径与原模型输出不一致"
                    item["first_batch_logits_max_difference"] = difference
                    checked_equivalence = True
                    del direct, cached_path

                features = spatial.cpu().numpy()
                for offset, index in enumerate(indices):
                    filename = dataset.samples[index].filename
                    stem = Path(filename).stem
                    feature_file = destination / f"{stem}.npy"
                    target_file = destination / f"{stem}_target.npz"
                    np.save(feature_file, features[offset], allow_pickle=False)
                    label = batch[offset][1][0].numpy().astype(np.uint8)
                    valid = batch[offset][2][0].numpy()
                    np.savez_compressed(target_file, label=label, valid=valid)
                    item["rows"].append({
                        "filename": filename,
                        "split": split,
                        "feature": str(feature_file.relative_to(ROOT)),
                        "feature_sha256": sha(feature_file),
                        "target": str(target_file.relative_to(ROOT)),
                        "target_sha256": sha(target_file),
                    })

                del images, hidden, tokens, spatial, features
                save()
                completed = min(start+4, len(dataset))
                if start % 40 == 0 or completed == len(dataset):
                    print(f"{split}：{completed}/{len(dataset)}", flush=True)

    assert len(item["rows"]) == 555
    item["status"] = "completed"
    save()
    print("缓存完成：", cache, flush=True)
    print(
        "首批缓存路径与原模型最大输出差：",
        item["first_batch_logits_max_difference"], flush=True,
    )
    del model
    torch.cuda.empty_cache()

record["status"] = "completed"
save()
print("\n两个版本缓存完成，索引：", OUT, flush=True)
