"""Frozen DINOv3 candidate-only engineering smoke, benchmark, and exploration."""
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
import transformers
from torch.utils.data import DataLoader
from src.models.dinov3_segmentation import build_local_dinov3
from src.training.exploratory import CandidateDataset, evaluate, save_checkpoint, load_checkpoint
from src.training.segmentation_steps import train_step

MANIFEST = ROOT / "data/interim/splits/moxitaidi_unified_spatial_manifest_v1.csv"
DATA = ROOT / "data/raw/CAS/Moxitaidi_UAV_0.6m"


def sha(path):
    with Path(path).open("rb") as f: return hashlib.file_digest(f, "sha256").hexdigest()


def state_hash(module):
    h = hashlib.sha256()
    for name, t in module.state_dict().items():
        h.update(name.encode()); h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--model-id", required=True, choices=("lvd1689m", "sat493m"))
    p.add_argument("--mode", choices=("smoke", "benchmark", "train"), default="smoke")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--crop", type=int, default=512)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--lr", type=float, default=.0001)
    a = p.parse_args()
    if a.batch_size < 1 or a.crop < 16 or a.crop % 16 or a.epochs < 1 or a.lr <= 0:
        p.error("invalid batch size, image size, epoch count or learning rate")
    if a.mode != "train" and a.epochs != 1: p.error("smoke and benchmark require one epoch")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(2); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    np.random.seed(a.seed); random.seed(a.seed)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    device=torch.device("cuda:0"); torch.cuda.reset_peak_memory_stats(device)
    run=Path(tempfile.mkdtemp(prefix="uav_dinov3_"+a.model_id+"_"+a.mode+"_",dir=ROOT/"experiments"))
    config={"model_id":a.model_id,"revision":a.snapshot.name,"snapshot":str(a.snapshot),
            "weight_sha256":sha(a.snapshot/"model.safetensors"),"seed":a.seed,
            "mode":a.mode,"batch_size":a.batch_size,"epochs":a.epochs,"crop":a.crop,
            "learning_rate":a.lr,"optimizer":"Adam","encoder":"frozen; eval mode",
            "decoder":"random Conv2d(1024,1,1) on patch tokens; bilinear align_corners=False",
            "threshold":.5,"loss":"masked BCE + batch-pooled soft Dice smooth=1 weights=1",
            "valid_policy":"all_pixels_provisional; NoData not validated",
            "sampling":"sorted filenames; shuffle seed+epoch","workers":0,
            "scope":"exploratory candidate-only; not formal E1 or independent evaluation"}
    processor=json.loads((a.snapshot/"preprocessor_config.json").read_text())
    config["normalization"]={"mean":processor["image_mean"],"std":processor["image_std"],
                             "rgb_scale":"uint8/255","resize":"disabled; 512x512 full image for train"}
    git=lambda *x:subprocess.check_output(["git","--no-optional-locks",*x],cwd=ROOT,text=True).strip()
    rec={"created_utc":datetime.now(timezone.utc).isoformat(),"status":"started","config":config,"epochs":[],
         "environment":{"torch":torch.__version__,"cuda_build":torch.version.cuda,"transformers":transformers.__version__,
                        "gpu":torch.cuda.get_device_name(0),"python":sys.version},
         "git_head":git("rev-parse","HEAD"),"git_status":git("status","--short","--untracked-files=all"),
         "source_sha256":{str(f.relative_to(ROOT)):sha(f) for f in [Path(__file__),ROOT/"src/models/dinov3_segmentation.py",ROOT/"src/training/exploratory.py",ROOT/"src/training/segmentation_steps.py",ROOT/"src/data/uav_data.py",ROOT/"src/metrics/binary_segmentation.py"]}}
    def save(): (run/"run.json").write_text(json.dumps(rec,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    save(); print("RUN",run,flush=True)
    try:
        train=CandidateDataset(MANIFEST,DATA,"train_candidate",limit=4 if a.mode=="smoke" else 16 if a.mode=="benchmark" else None,crop=a.crop)
        val=CandidateDataset(MANIFEST,DATA,"val_candidate",crop=a.crop) if a.mode=="train" else None
        rec["manifest_sha256"]=train.samples[0].manifest_sha256
        rec["train_samples"]=[s.filename for s in train.samples]
        rec["val_samples"]=[s.filename for s in val.samples] if val else []
        for split in ("train","val"):
            rec[split+"_samples_sha256"]=hashlib.sha256(("\n".join(rec[split+"_samples"])+"\n").encode()).hexdigest()
        model=build_local_dinov3(a.snapshot).to(device)
        opt=torch.optim.Adam(model.head.parameters(),lr=a.lr)
        rec["parameters"]={"total":sum(t.numel() for t in model.parameters()),
                           "trainable":sum(t.numel() for t in model.parameters() if t.requires_grad)}
        rec["encoder_state_sha256_before"]=state_hash(model.encoder) if a.mode=="smoke" else None
        best=-1.
        for epoch in range(a.epochs):
            loader=DataLoader(train,batch_size=a.batch_size,shuffle=a.mode=="train",
                              generator=torch.Generator().manual_seed(a.seed+epoch),num_workers=0)
            start=time.perf_counter(); total=count=0; old=model.head.weight.detach().clone()
            for x,y,v in loader:
                x,y,v=x.to(device),y.to(device),v.to(device)
                n=int(v.sum()); total+=train_step(model,opt,x,y,v,loss_kind="dice_bce")*n; count+=n
            torch.cuda.synchronize()
            item={"epoch":epoch+1,"train_loss":total/count,"train_seconds":time.perf_counter()-start,
                  "head_changed":not torch.equal(old,model.head.weight),
                  "encoder_training":model.encoder.training,"peak_allocated_bytes":torch.cuda.max_memory_allocated(device)}
            if not item["head_changed"] or item["encoder_training"]: raise RuntimeError("update or frozen mode check failed")
            if val:
                t=time.perf_counter()
                item["validation"]=evaluate(model,DataLoader(val,batch_size=a.batch_size,shuffle=False),device,loss_kind="dice_bce")
                torch.cuda.synchronize(); item["validation_seconds"]=time.perf_counter()-t
                score=item["validation"]["confusion"]["metrics"]["iou"]
                if score is not None and score>best:
                    best=score; rec["best_epoch"]=epoch+1
                    save_checkpoint(run/"best.pt",model,opt,epoch+1,config,best)
                save_checkpoint(run/"last.pt",model,opt,epoch+1,config,best)
            rec["epochs"].append(item); save(); print(json.dumps(item),flush=True)
        if a.mode=="smoke":
            after=state_hash(model.encoder); rec["encoder_state_sha256_after"]=after
            if after!=rec["encoder_state_sha256_before"]: raise RuntimeError("frozen encoder state changed")
            head=model.head.weight.detach().clone()
            opt_before={k:v.clone() if torch.is_tensor(v) else v for k,v in opt.state[model.head.weight].items()}
            save_checkpoint(run/"smoke_checkpoint.pt",model,opt,1,config,0.)
            with torch.no_grad(): model.head.weight.add_(1)
            load_checkpoint(run/"smoke_checkpoint.pt",model,opt,device)
            restored=all(torch.equal(v,opt.state[model.head.weight][k]) if torch.is_tensor(v) else v==opt.state[model.head.weight][k] for k,v in opt_before.items())
            rec["checkpoint_restore"]={"head":bool(torch.equal(head,model.head.weight)),"optimizer":restored,
                                       "encoder":state_hash(model.encoder)==after}
            if not all(rec["checkpoint_restore"].values()): raise RuntimeError("checkpoint restoration failed")
        rec["status"]="completed"
    except Exception as e:
        rec["status"]="failed"; rec["error_type"]=type(e).__name__; rec["error"]=str(e)[:500]
        raise
    finally:
        rec["finished_utc"]=datetime.now(timezone.utc).isoformat(); save()
    print("COMPLETED",run,flush=True)


if __name__=="__main__": main()
