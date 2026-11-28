"""Offline GPU checks using pretrained encoders and synthetic inputs only."""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import transformers
import huggingface_hub
from src.models.dinov3_segmentation import build_local_dinov3
from src.training.segmentation_steps import train_step

OUT = ROOT / "reports/tables/dinov3_gpu_synthetic_v1.json"
if OUT.exists():
    raise SystemExit("检查记录已存在，未重复运行。")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable")

torch.set_num_threads(2)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
weights = ROOT / "weights/dinov3"
checks = json.loads((weights / "checksums.json").read_text(encoding="utf-8"))
versions = json.loads(
    (weights / "download_record.json").read_text(encoding="utf-8")
)

report = {
    "scope": "Synthetic GPU functionality only; no project imagery",
    "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "model_source_sha256": hashlib.sha256(
        (ROOT / "src/models/dinov3_segmentation.py").read_bytes()
    ).hexdigest(),
    "environment": {
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "gpu": torch.cuda.get_device_name(0),
    },
    "results": [],
}

for name in (
    "dinov3-vitl16-pretrain-lvd1689m",
    "dinov3-vitl16-pretrain-sat493m",
):
    print("\n检查：", name, flush=True)
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    model = build_local_dinov3(weights / name).cuda()
    encoder_parameters = list(model.encoder.parameters())
    assert encoder_parameters
    assert all(not p.requires_grad for p in encoder_parameters), \
        "编码器未完全冻结"

    model.train()
    assert not model.encoder.training, \
        "model.train()后编码器进入训练模式，需要先修复封装"

    parameters = [p for p in model.parameters() if p.requires_grad]
    assert parameters, "没有可训练解码头参数"
    optimizer = torch.optim.Adam(parameters, lr=0.0001)
    before = [p.detach().clone() for p in parameters]

    image = torch.rand((1, 3, 512, 512), device="cuda")
    target = torch.zeros((1, 1, 512, 512), device="cuda")
    target[:, :, 128:384, 128:384] = 1
    valid = torch.ones_like(target, dtype=torch.bool)
    valid[:, :, :32, :32] = False

    loss = train_step(
        model, optimizer, image, target, valid,
        loss_kind="dice_bce",
        dice_smooth=1.0, bce_weight=1.0, dice_weight=1.0,
    )
    assert any(
        not torch.equal(old, new.detach())
        for old, new in zip(before, parameters)
    ), "解码头参数没有更新"
    assert all(p.grad is None for p in encoder_parameters), \
        "冻结编码器出现梯度"
    assert not model.encoder.training, "训练步骤改变了编码器模式"

    model.eval()
    with torch.no_grad():
        logits = model(image)
    assert tuple(logits.shape) == (1, 1, 512, 512)
    assert bool(torch.isfinite(logits).all())
    torch.cuda.synchronize()

    item = {
        "model": name,
        "revision": versions[name]["revision"],
        "weight_sha256": checks[f"{name}/model.safetensors"]["sha256"],
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in parameters),
        "encoder_frozen": True,
        "encoder_eval_during_training": True,
        "encoder_gradients_absent": True,
        "decoder_updated": True,
        "synthetic_input_shape": list(image.shape),
        "output_shape": list(logits.shape),
        "loss": loss,
        "finite_logits": True,
        "seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    report["results"].append(item)
    print(json.dumps(item, ensure_ascii=False, indent=2), flush=True)

    del model, optimizer, parameters, before, encoder_parameters
    del image, target, valid, logits
    torch.cuda.empty_cache()

report["status"] = "completed"
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("x", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    f.write("\n")
print("\n两个版本检查完成：", OUT)
