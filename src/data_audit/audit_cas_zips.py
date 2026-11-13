#!/usr/bin/env python3
"""Read-only CAS ZIP inventory and explicit sampled image/mask checks."""
import argparse
import csv
import io
import json
import random
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

TIFF_EXT = {'.tif', '.tiff'}


def write_csv(path, rows, columns):
    with path.open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cas-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--sample', type=int, default=30,
                        help='maximum paired images per archive; 0 means all (default: 30)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.sample < 0:
        parser.error('--sample must be >= 0')
    archives = sorted(args.cas_dir.glob('*.zip'))
    if not archives:
        raise SystemExit(f'No ZIP files found: {args.cas_dir}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / 'cas_zip_inventory.csv'
    sample_path = args.output_dir / 'cas_sample_checks.csv'
    meta_path = args.output_dir / 'cas_audit_metadata.json'
    for path in (summary_path, sample_path, meta_path):
        if path.exists():
            raise SystemExit(f'Output exists, select a new output directory: {path}')

    rng = random.Random(args.seed)
    summaries, checks = [], []
    for archive in archives:
        print(f'Checking {archive.name}', flush=True)
        result = dict(archive=archive.name, image_count=0, mask_count=0,
                      label_count=0, matched_pairs=0, missing_mask=0,
                      extra_mask=0, duplicate_keys=0, sampled_pairs=0,
                      sample_errors=0, status='ok', detail='')
        try:
            with zipfile.ZipFile(archive) as z:
                groups = {'img': {}, 'mask': {}, 'label': {}}
                duplicates = []
                for entry in z.infolist():
                    if entry.is_dir():
                        continue
                    path = PurePosixPath(entry.filename.replace('\\', '/'))
                    if path.suffix.lower() not in TIFF_EXT or len(path.parts) < 2:
                        continue
                    kind = path.parts[-2].lower()
                    if kind not in groups:
                        continue
                    key = (str(PurePosixPath(*path.parts[:-2])), path.stem)
                    if key in groups[kind]:
                        duplicates.append(f'{kind}:{key}')
                    else:
                        groups[kind][key] = entry
                for kind, field in [('img','image_count'),('mask','mask_count'),('label','label_count')]:
                    result[field] = len(groups[kind])
                images, masks = set(groups['img']), set(groups['mask'])
                pairs = sorted(images & masks)
                result.update(matched_pairs=len(pairs), missing_mask=len(images-masks),
                              extra_mask=len(masks-images), duplicate_keys=len(duplicates))
                if not images and not masks:
                    result.update(status='no_recognized_tiff_folders', detail='check archive layout manually')
                elif duplicates or images != masks:
                    result.update(status='needs_review', detail='duplicate names or unmatched img/mask; inspect layout')
                chosen = pairs if args.sample == 0 else sorted(rng.sample(pairs, min(len(pairs), args.sample)))
                for key in chosen:
                    item = dict(archive=archive.name, image=groups['img'][key].filename,
                                mask=groups['mask'][key].filename, width='', height='',
                                channels='', image_dtype='', mask_width='', mask_height='',
                                mask_bands='', mask_dtype='', mask_values='', landslide_fraction='', status='ok', detail='')
                    try:
                        with Image.open(io.BytesIO(z.read(groups['img'][key]))) as im:
                            item['width'], item['height'] = im.size
                            item['channels'] = len(im.getbands())
                            item['image_dtype'] = str(np.asarray(im).dtype)
                        with Image.open(io.BytesIO(z.read(groups['mask'][key]))) as m:
                            item['mask_width'], item['mask_height'] = m.size
                            item['mask_bands'] = len(m.getbands())
                            mask = np.asarray(m)
                            item['mask_dtype'] = str(mask.dtype)
                        vals = np.unique(mask)
                        item['mask_values'] = ' '.join(map(str, vals[:32]))
                        if len(vals) > 32:
                            item['mask_values'] += ' ...'
                        if item['mask_bands'] == 1 and np.isin(vals, [0,1]).all():
                            item['landslide_fraction'] = float(np.mean(mask == 1))
                        problems = []
                        if (item['width'], item['height']) != (512,512): problems.append('image_size')
                        if (item['width'], item['height']) != (item['mask_width'],item['mask_height']): problems.append('size_mismatch')
                        if item['channels'] != 3: problems.append('image_channels')
                        if item['mask_bands'] != 1 or not np.isin(vals,[0,1]).all(): problems.append('mask_not_0_1_single_band')
                        if problems: item.update(status='needs_review', detail=';'.join(problems))
                    except Exception as e:
                        item.update(status='read_error', detail=f'{type(e).__name__}: {e}')
                    checks.append(item)
                    result['sampled_pairs'] += 1
                    result['sample_errors'] += int(item['status'] != 'ok')
                if result['sample_errors'] and result['status'] == 'ok':
                    result['status'] = 'sample_needs_review'
        except (OSError, zipfile.BadZipFile, RuntimeError) as e:
            result.update(status='archive_error', detail=f'{type(e).__name__}: {e}')
        summaries.append(result)
    write_csv(summary_path, summaries, list(summaries[0]))
    fields=['archive','image','mask','width','height','channels','image_dtype','mask_width','mask_height','mask_bands','mask_dtype','mask_values','landslide_fraction','status','detail']
    write_csv(sample_path, checks, fields)
    metadata={'cas_dir':str(args.cas_dir),'archives':len(archives),'sample_per_archive':args.sample,
              'seed':args.seed,'scope':'ZIP entry names inventoried; image and mask pixels sampled only. No ZIP integrity, ground resolution, no-data, leakage, or georeferencing verification.',
              'selection':'sorted matches; Python random.sample sequentially using seeded RNG; sample=0 selects all',
              'statuses':dict(Counter(s['status'] for s in summaries))}
    with meta_path.open('x',encoding='utf-8') as f:
        json.dump(metadata,f,ensure_ascii=False,indent=2)
    print('Done:', dict(Counter(s['status'] for s in summaries)))
    print('Saved:', summary_path, sample_path, meta_path, sep='\n  ')


if __name__ == '__main__':
    main()
