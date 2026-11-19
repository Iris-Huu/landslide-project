"""Read only the UAV component-2 train/val candidates; produce new JSON and CSV reports."""
import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import warnings

import numpy as np
# Avoid GDAL creating auxiliary files next to raw inputs.
os.environ["GDAL_PAM_ENABLED"] = "NO"
import rasterio
from rasterio.errors import NotGeoreferencedWarning
from affine import Affine

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / 'data/interim/splits/moxitaidi_unified_spatial_manifest_v1.csv'
AUDIT = ROOT / 'reports/tables/moxitaidi_nodata_audit_v1.csv'
DATA = ROOT / 'data/raw/CAS/Moxitaidi_UAV_0.6m'
SPLITS = ('train_candidate', 'val_candidate')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metadata(src):
    gcps, gcp_crs = src.gcps
    rpcs = src.rpcs
    return {'width': src.width, 'height': src.height, 'channels': src.count,
            'dtypes': list(src.dtypes), 'crs': str(src.crs) if src.crs else None,
            'transform': list(src.transform)[:6], 'identity_transform': src.transform == Affine.identity(),
            'gcp_count': len(gcps), 'gcp_crs': str(gcp_crs) if gcp_crs else None,
            'has_rpc': rpcs is not None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary', type=Path, required=True)
    p.add_argument('--details', type=Path, required=True)
    args = p.parse_args()
    if args.summary == args.details or args.summary.exists() or args.details.exists():
        raise SystemExit('Output paths must be distinct and not already exist')
    started = datetime.now(timezone.utc).isoformat()
    with MANIFEST.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        required = {'platform', 'component', 'split', 'filename'}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError('Missing required manifest fields')
        selected = [r for r in reader if r['platform'] == 'UAV' and r['split'] in SPLITS]
    if not selected or any(r['component'] != '2' for r in selected):
        raise ValueError('Empty selection or train/val candidate outside component 2')
    names = [r['filename'] for r in selected]
    if len(names) != len(set(names)) or any(not n or Path(n).name != n or n in ('.', '..') for n in names):
        raise ValueError('Duplicate or unsafe selected filename')
    audit = {}
    with AUDIT.open(encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            if r['subset'] == 'Moxitaidi_UAV_0.6m' and r['filename'] in names:
                if r['filename'] in audit:
                    raise ValueError('Duplicate existing NoData audit row: ' + r['filename'])
                audit[r['filename']] = r
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', NotGeoreferencedWarning)
        for i, sample in enumerate(selected, 1):
            name = sample['filename']
            row = {'split': sample['split'], 'filename': name, 'image': None, 'mask': None,
                   'image_error': None, 'mask_error': None, 'issues': [], 'mask_values': None,
                   'landslide_percent_full_image': None, 'dimensions_match': None, 'georeferencing_comparison': None}
            for kind, folder in (('image', 'img'), ('mask', 'mask')):
                try:
                    with rasterio.open(DATA / folder / name) as src:
                        meta = metadata(src)
                        pixels = src.read()  # force complete decoding of every band
                        row[kind] = meta
                        if kind == 'mask':
                            vals = np.unique(pixels)
                            row['mask_values'] = [v.item() if np.isfinite(v) else str(v) for v in vals]
                            if src.count != 1:
                                row['issues'].append('mask_channels_not_one')
                            if not np.isin(vals, [0, 1]).all():
                                row['issues'].append('mask_values_not_binary_0_1')
                            if src.count == 1 and np.isin(vals, [0, 1]).all():
                                row['landslide_percent_full_image'] = float(np.mean(pixels[0] == 1) * 100)
                except Exception as exc:
                    row[kind + '_error'] = f'{type(exc).__name__}: {exc}'
                    row['issues'].append(kind + '_read_error')
            im, ma = row['image'], row['mask']
            if im and ma:
                row['dimensions_match'] = (im['width'], im['height']) == (ma['width'], ma['height'])
                if not row['dimensions_match']:
                    row['issues'].append('dimensions_mismatch')
                if im['channels'] != 3:
                    row['issues'].append('image_channels_not_three')
                im_ref = bool(im['crs'] or im['gcp_count'] or im['has_rpc'] or not im['identity_transform'])
                ma_ref = bool(ma['crs'] or ma['gcp_count'] or ma['has_rpc'] or not ma['identity_transform'])
                if not im_ref and not ma_ref:
                    row['georeferencing_comparison'] = 'neither_has_georeference'
                elif im['crs'] and ma['crs'] and not im['gcp_count'] and not ma['gcp_count'] and not im['has_rpc'] and not ma['has_rpc']:
                    row['georeferencing_comparison'] = 'affine_crs_equal' if (im['crs'], im['transform']) == (ma['crs'], ma['transform']) else 'affine_crs_mismatch'
                else:
                    row['georeferencing_comparison'] = 'incomplete_or_non_affine_requires_review'
                if row['georeferencing_comparison'] in ('affine_crs_mismatch', 'incomplete_or_non_affine_requires_review'):
                    row['issues'].append('georeference_requires_review')
            rows.append(row)
            if i % 100 == 0 or i == len(selected):
                print(f'{i}/{len(selected)} checked', flush=True)
    summary = {'checked_at_utc': started, 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
               'environment': {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(), 'numpy': np.__version__, 'rasterio': rasterio.__version__, 'gdal': rasterio.__gdal_version__},
               'inputs': {str(path.relative_to(ROOT)): {'sha256': sha(path)} for path in (MANIFEST, AUDIT)},
               'scope': 'UAV component 2 train_candidate and val_candidate only; full image and mask pixel reads',
               'limitations': ['Full-image landslide ratios do not exclude unverified invalid areas.', 'NoData counts are previous audit results, not recalculated or a validated valid-pixel mask.', 'Readable pixels, equal dimensions and metadata do not prove physical registration or correct labels.', 'No test, buffer, unassigned or SAT imagery/masks were read.'],
               'splits': {}}
    for split in SPLITS:
        group = [r for r in rows if r['split'] == split]
        ratios = [r['landslide_percent_full_image'] for r in group if r['landslide_percent_full_image'] is not None]
        old = [audit[r['filename']] for r in group if r['filename'] in audit]
        stats = lambda field: dict(Counter(json.dumps(r[field], sort_keys=True) for r in group))
        summary['splits'][split] = {'samples': len(group), 'image_read_failures': sum(r['image_error'] is not None for r in group),
            'mask_read_failures': sum(r['mask_error'] is not None for r in group), 'samples_with_issues': sum(bool(r['issues']) for r in group),
            'issue_counts': dict(Counter(x for r in group for x in r['issues'])),
            'issue_details': [{'filename': r['filename'], 'issues': r['issues'], 'image_error': r['image_error'], 'mask_error': r['mask_error']} for r in group if r['issues']],
            'georeferencing_comparison': dict(Counter(r['georeferencing_comparison'] or 'not_comparable_read_failure' for r in group)),
            'full_image_landslide_percent_per_image': {'count': len(ratios), 'mean': float(np.mean(ratios)) if ratios else None,
                'min': min(ratios) if ratios else None, 'p25': float(np.percentile(ratios, 25)) if ratios else None,
                'median': float(np.median(ratios)) if ratios else None, 'p75': float(np.percentile(ratios, 75)) if ratios else None,
                'max': max(ratios) if ratios else None, 'zero_landslide_images': sum(x == 0 for x in ratios)},
            'existing_suspected_nodata_audit': {'matched_rows': len(old), 'missing_rows': len(group)-len(old),
                'suspected_pixels': sum(int(r['candidate_pixels']) for r in old), 'overlap_with_positive_pixels': sum(int(r['overlap_pixels']) for r in old),
                'images_with_suspected_pixels': sum(int(r['candidate_pixels']) > 0 for r in old),
                'images_with_positive_overlap': sum(int(r['overlap_pixels']) > 0 for r in old)}}
    with args.details.open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, allow_nan=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    with args.summary.open('x', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(summary['splits'], ensure_ascii=False, indent=2))
    return int(any(r['issues'] for r in rows))

if __name__ == '__main__':
    raise SystemExit(main())
