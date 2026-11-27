"""UAV v1 training/validation dataset; test loading is deliberately excluded."""
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.nodata_v1 import valid_mask_v1
from src.data.uav_data import Sample, read_sample


class FormalUAVDataset(Dataset):
    def __init__(self, manifest, data_dir, split):
        if split not in ("train", "val"):
            raise ValueError("Only train and val are allowed here")

        path = Path(manifest).resolve()
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()

        metadata = json.loads(
            path.with_suffix(".json").read_text(encoding="utf-8")
        )
        if digest != metadata["formal_manifest_sha256"]:
            raise ValueError("Frozen manifest SHA-256 mismatch")

        rows = list(csv.DictReader(
            io.StringIO(payload.decode("utf-8-sig"))
        ))
        expected = {
            "train": 484, "val": 71, "test": 149,
            "buffer": 57, "excluded": 223,
        }
        if Counter(r["split_v1"] for r in rows) != Counter(expected):
            raise ValueError("Frozen split counts do not match protocol")

        names = [r["filename"] for r in rows]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate filenames")

        for row in rows:
            name = row["filename"]
            if (
                not name or Path(name).name != name
                or "/" in name or "\\" in name or name in (".", "..")
            ):
                raise ValueError("Unsafe filename")
            if row["platform"] != "UAV":
                raise ValueError("Non-UAV record")
            if row["split_v1"] != "excluded" and row["component"] != "2":
                raise ValueError("Assigned record outside component 2")

        self.rows = sorted(
            (r for r in rows if r["split_v1"] == split),
            key=lambda r: r["filename"],
        )
        self.manifest_sha256 = digest
        self.split = split
        self.data_dir = Path(data_dir)

        # Reuse the validated TIFF reader without changing legacy experiments.
        candidate_split = {"train": "train_candidate", "val": "val_candidate"}[split]
        self.samples = [
            Sample(
                r["filename"], candidate_split, "2",
                str(path), digest,
            )
            for r in self.rows
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = read_sample(self.samples[index], self.data_dir)

        image = item["image"]
        label = item["mask"]
        if image.shape != (3, 512, 512) or label.shape != (512, 512):
            raise ValueError("Formal v1 requires full 512x512 tiles")

        raw = np.rint(image * np.float32(255)).astype(np.uint8)
        valid = valid_mask_v1(raw, label)

        if np.any((~valid) & (label == 1)):
            raise ValueError("Positive label incorrectly ignored")

        return (
            torch.from_numpy(image.copy()),
            torch.from_numpy(label.copy()).unsqueeze(0).float(),
            torch.from_numpy(valid.copy()).unsqueeze(0),
        )
