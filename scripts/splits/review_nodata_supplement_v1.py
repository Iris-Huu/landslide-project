"""Supplement train/val-only unverified NoData visual review; never overwrite."""
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from PIL import Image
from src.data.uav_data import select_samples, read_sample
from src.data_audit.audit_nodata_candidate import find_candidate
from src.metrics.binary_segmentation import ConfusionMatrix

MANIFEST = ROOT / "data/interim/splits/moxitaidi_unified_spatial_manifest_v1.csv"
AUDIT = ROOT / "reports/tables/moxitaidi_nodata_audit_v1.csv"
OLD_REVIEW = ROOT / "reports/tables/uav_train_val_review_selection_v1.json"
SENSITIVITY = ROOT / "experiments/uav_val_error_y69erm3a/summary.json"
DATA = ROOT / "data/raw/CAS/Moxitaidi_UAV_0.6m"
OUT = ROOT / "reports/figures/uav_nodata_review_supplement_v1"
TABLE = ROOT / "reports/tables/uav_nodata_review_supplement_v1.json"


def validate_partition(full, retained, excluded):
    keys = ("tp", "fp", "fn", "tn")
    diff = {k: full[k] - retained[k] for k in keys}
    if any(v < 0 for v in diff.values()) or sum(diff.values()) != excluded["total"]:
        raise ValueError("inconsistent excluded confusion matrix")
    if diff["tp"] + diff["fn"] != excluded["positive"] or diff["fp"] != excluded["fp"]:
        raise ValueError("inconsistent excluded label or false-positive counts")
    for counts in (full, retained):
        expected = ConfusionMatrix(**{k: counts[k] for k in keys}).result()
        if expected["metrics"] != counts["metrics"] or expected["valid_pixels"] != counts["valid_pixels"]:
            raise ValueError("inconsistent saved metrics")
    return diff


def main():
    if OUT.exists() or TABLE.exists():
        raise FileExistsError("Refusing to overwrite previous review output")
    with AUDIT.open(encoding="utf-8-sig", newline="") as f:
        audit = {r["filename"]: r for r in csv.DictReader(f) if r["subset"] == "Moxitaidi_UAV_0.6m"}
    previous = json.loads(OLD_REVIEW.read_text())
    details = []
    choices = []
    summary = {}
    for split in ("train_candidate", "val_candidate"):
        samples = select_samples(MANIFEST, split, "2")
        by_name = {s.filename: s for s in samples}
        if any(s.filename not in audit for s in samples):
            raise ValueError("sample missing from prior audit")
        rows = [audit[s.filename] for s in samples]
        summary[split] = {"samples": len(rows),
            "images_with_suspected": sum(int(r["candidate_pixels"]) > 0 for r in rows),
            "images_with_positive_overlap": sum(int(r["overlap_pixels"]) > 0 for r in rows),
            "suspected_pixels": sum(int(r["candidate_pixels"]) for r in rows),
            "overlap_positive_pixels": sum(int(r["overlap_pixels"]) for r in rows)}
        for reason in ("no_suspected_pixels", "largest_positive_overlap"):
            old = next(x for x in previous["figures"] if x["split"] == split and x["selection_reason"] == reason)
            if old["filename"] not in by_name:
                raise ValueError("previous selected sample outside permitted split")
            with Image.open(ROOT / old["output"]) as im:
                im.verify()
            details.append(dict(old, reused=True))
        eligible = [r for r in rows if int(r["candidate_pixels"]) > 0 and int(r["overlap_pixels"]) == 0]
        selected = sorted(eligible, key=lambda r: (-int(r["candidate_pixels"]), r["filename"]))[0]
        choices.append((by_name[selected["filename"]], selected))
    OUT.mkdir(parents=True, exist_ok=False)
    for sample, prior in choices:
        data = read_sample(sample, DATA)
        raw = np.rint(data["image"] * 255).astype(np.uint8)
        suspect = find_candidate(raw)
        label = data["mask"].astype(bool)
        overlap = suspect & label
        if int(suspect.sum()) != int(prior["candidate_pixels"]) or int(overlap.sum()) != int(prior["overlap_pixels"]):
            raise ValueError(f"{sample.filename}: historical audit disagrees")
        rgb = data["image"].transpose(1, 2, 0)
        fig, ax = plt.subplots(1, 4, figsize=(17, 5), constrained_layout=True)
        ax[0].imshow(rgb); ax[0].set_title("RGB image")
        ax[1].imshow(label, cmap="gray", vmin=0, vmax=1); ax[1].set_title("Label: white = positive")
        ax[2].imshow(suspect, cmap="gray", vmin=0, vmax=1); ax[2].set_title("Suspected: UNVERIFIED")
        ax[3].imshow(rgb)
        overlay = np.zeros((*label.shape, 4))
        overlay[label & ~suspect] = [1, 0, 0, .55]
        overlay[suspect & ~label] = [1, .85, 0, .65]
        overlay[overlap] = [1, 0, 1, .85]
        ax[3].imshow(overlay); ax[3].set_title("Label / suspected / overlap")
        for axis in ax: axis.axis("off")
        fig.suptitle(f"{sample.filename} | {sample.split} | highest suspected pixels without overlap")
        fig.legend(handles=[Patch(color="red", label="Positive label"), Patch(color="gold", label="Unverified suspected"), Patch(color="magenta", label="Overlap")], loc="lower center", ncol=3)
        path = OUT / f"{sample.split}_{Path(sample.filename).stem}.png"
        fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)
        with Image.open(path) as im: im.verify()
        details.append({"filename":sample.filename,"split":sample.split,"selection_reason":"highest_suspected_without_positive_overlap",
                        "suspected_pixels":int(suspect.sum()),"overlap_pixels":int(overlap.sum()),
                        "output":str(path.relative_to(ROOT)),"reused":False})
    old = json.loads(SENSITIVITY.read_text())
    full = old["models"]["weighted"]["regions"]["direct_center"]
    sens = old["sensitivity_excluding_unverified_suspected"]
    diff = validate_partition(full, sens["counts"], sens["excluded"])
    checkpoint = ROOT / old["models"]["weighted"]["checkpoint"]
    with checkpoint.open("rb") as f: checkpoint_hash = hashlib.file_digest(f,"sha256").hexdigest()
    if checkpoint_hash != old["models"]["weighted"]["checkpoint_sha256"]:
        raise ValueError("historical checkpoint hash mismatch")
    report = {"created_utc":datetime.now(timezone.utc).isoformat(),
        "scope":"UAV component 2 train_candidate and val_candidate only",
        "manifest_sha256":hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "audit_sha256":hashlib.sha256(AUDIT.read_bytes()).hexdigest(),
        "rules":"per split: maximum positive overlap; first filename without suspected; maximum suspected with zero positive overlap; filename ascending for ties",
        "suspected_rule":"4-connected edge-connected RGB all-0 or all-255, unverified",
        "historical_audit_summary":summary,"figures":details,
        "sensitivity":{"source":str(SENSITIVITY.relative_to(ROOT)),"source_sha256":hashlib.sha256(SENSITIVITY.read_bytes()).hexdigest(),
                       "checkpoint":str(checkpoint.relative_to(ROOT)),"checkpoint_sha256":checkpoint_hash,
                       "scope":"weighted U-Net best, val_candidate direct 256 center crop, threshold 0.5; NOT DeepLab or SegFormer full image",
                       "all_pixels":full,"retained":sens["counts"],"excluded":sens["excluded"],"excluded_confusion":diff,
                       "excluded_fraction":sens["excluded"]["total"]/full["valid_pixels"]},
        "limitations":["Suspected NoData not validated; not a final valid mask", "Overlap does not prove incorrect label or NoData",
                       "Visual inspection cannot establish alignment or overall label quality",
                       "Validation was used for model development and selection; not independent evaluation"]}
    with TABLE.open("x", encoding="utf-8") as f: json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")
    print(json.dumps({"audit":summary,"new_figures":details[-2:],"sensitivity":report["sensitivity"]},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
