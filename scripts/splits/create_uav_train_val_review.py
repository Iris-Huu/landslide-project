"""Create four reproducible train/val visual inspections; never access other splits."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from PIL import Image
from src.data.uav_data import select_samples, read_sample
from src.data_audit.audit_nodata_candidate import find_candidate

MANIFEST = ROOT / 'data/interim/splits/moxitaidi_unified_spatial_manifest_v1.csv'
AUDIT = ROOT / 'reports/tables/moxitaidi_nodata_audit_v1.csv'
DATA = ROOT / 'data/raw/CAS/Moxitaidi_UAV_0.6m'
OUT = ROOT / 'reports/figures/uav_train_val_review_v1'
REPORT = ROOT / 'reports/tables/uav_train_val_review_selection_v1.json'


def main():
    if OUT.exists() or REPORT.exists():
        raise SystemExit('Output directory or report already exists; refusing overwrite')
    audit = {}
    with AUDIT.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            if row['subset'] == 'Moxitaidi_UAV_0.6m':
                audit[row['filename']] = row
    selections, smoke = [], []
    for split in ('train_candidate', 'val_candidate'):
        samples = select_samples(MANIFEST, split, '2')
        by_name = {s.filename: s for s in samples}
        # Basic reproducible real-data smoke check, separate from selection of figures.
        for sample in sorted(samples, key=lambda s: s.filename)[:3]:
            data = read_sample(sample, DATA)
            assert data['image'].shape == (3, 512, 512) and data['image'].dtype == np.float32
            assert data['mask'].shape == (512, 512) and data['mask'].dtype == np.uint8
            assert data['valid_mask'].dtype == np.bool_ and data['valid_mask'].all()
            assert np.min(data['image']) >= 0 and np.max(data['image']) <= 1
            smoke.append({'split': split, 'filename': sample.filename, 'result': 'passed'})
        eligible = [s for s in samples if s.filename in audit]
        plain = sorted((s for s in eligible if int(audit[s.filename]['candidate_pixels']) == 0), key=lambda s: s.filename)
        overlap = sorted((s for s in eligible if int(audit[s.filename]['overlap_pixels']) > 0), key=lambda s: (-int(audit[s.filename]['overlap_pixels']), s.filename))
        if not plain or not overlap:
            raise RuntimeError(f'{split}: insufficient audited examples for both categories')
        selections += [(plain[0], 'no_suspected_pixels'), (overlap[0], 'largest_positive_overlap')]
    OUT.mkdir(parents=True)
    details = []
    for sample, reason in selections:
        data = read_sample(sample, DATA)
        rgb = data['image'].transpose(1, 2, 0)
        mask = data['mask'].astype(bool)
        raw = np.rint(data['image'] * 255).astype(np.uint8)
        suspicious = find_candidate(raw)
        old = audit[sample.filename]
        if int(suspicious.sum()) != int(old['candidate_pixels']) or int((suspicious & mask).sum()) != int(old['overlap_pixels']):
            raise RuntimeError(f'{sample.filename}: recomputed suspected region disagrees with existing audit')
        fig, axes = plt.subplots(1, 4, figsize=(17, 5), constrained_layout=True)
        axes[0].imshow(rgb); axes[0].set_title('RGB image')
        axes[1].imshow(mask, cmap='gray', vmin=0, vmax=1, interpolation='nearest'); axes[1].set_title('Label mask: white = landslide')
        axes[2].imshow(rgb); overlay = np.zeros((*mask.shape, 4)); overlay[mask] = [1, 0, 0, .5]; axes[2].imshow(overlay); axes[2].set_title('Red = landslide label')
        axes[3].imshow(rgb); overlay2 = np.zeros((*mask.shape, 4)); overlay2[mask] = [1, 0, 0, .50]; overlay2[suspicious] = [1, .85, 0, .65]; overlay2[mask & suspicious] = [1, 0, 1, .85]; axes[3].imshow(overlay2); axes[3].set_title('Suspected region: UNVERIFIED')
        for ax in axes: ax.axis('off')
        fig.suptitle(f'{sample.filename} | {sample.split} | {reason}', fontsize=14)
        fig.legend(handles=[Patch(color='red', label='Landslide label'), Patch(color='gold', label='Suspected region (unverified)'), Patch(color='magenta', label='Overlap')], loc='lower center', ncol=3, fontsize=11)
        path = OUT / f'{sample.split}_{reason}_{Path(sample.filename).stem}.png'
        fig.savefig(path, dpi=140, bbox_inches='tight'); plt.close(fig)
        with Image.open(path) as im:
            im.verify()
        details.append({'filename': sample.filename, 'split': sample.split, 'selection_reason': reason,
                        'historical_suspected_pixels': int(old['candidate_pixels']), 'historical_positive_overlap_pixels': int(old['overlap_pixels']),
                        'output': str(path.relative_to(ROOT)), 'source_manifest_sha256': sample.manifest_sha256})
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'only UAV component 2 train_candidate and val_candidate',
              'selection_rule': 'Per split, lexicographically first with candidate_pixels=0; largest overlap_pixels>0 with filename tie-break.',
              'suspected_rule': 'Existing find_candidate: 4-connected edge-connected all-channel pure 0 or 255; unverified.',
              'audit_path': str(AUDIT.relative_to(ROOT)), 'audit_sha256': hashlib.sha256(AUDIT.read_bytes()).hexdigest(),
              'colors': {'red': 'label', 'yellow': 'unverified suspected region', 'magenta': 'label and suspected overlap'},
              'parameters': {'dpi': 140, 'label_alpha': .5, 'suspected_alpha': .65, 'overlap_alpha': .85},
              'real_data_smoke_checks': smoke, 'figures': details,
              'limitations': ['No georeference; visual inspection cannot prove geographic registration.', 'Suspected areas are not validated as invalid pixels.', 'Outputs include imagery and are ignored by git under reports/figures.']}
    with REPORT.open('x', encoding='utf-8') as f: json.dump(report, f, indent=2, ensure_ascii=False); f.write('\n')
    print(json.dumps(report, indent=2, ensure_ascii=False))

if __name__ == '__main__': main()
