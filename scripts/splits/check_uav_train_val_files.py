#!/usr/bin/env python3
"""Check only UAV train/validation candidate metadata and file existence; never open imagery."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ('train_candidate', 'val_candidate')
MANIFEST = ROOT / 'data/interim/splits/moxitaidi_unified_spatial_manifest_v1.csv'
CANDIDATE = ROOT / 'data/interim/splits/moxitaidi_uav_split_candidate_v1.csv'
VALIDATION = ROOT / 'reports/tables/uav_split_validation_recheck_v1.json'
DATA = ROOT / 'data/raw/CAS/Moxitaidi_UAV_0.6m'

def read_csv(path, required):
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or not required.issubset(fields):
            raise ValueError(f'{path}: duplicate or missing fields: {fields}')
        rows = [r for r in reader if r.get('split') in SPLITS and (path != MANIFEST or r.get('platform') == 'UAV')]
    return fields, rows

def duplicates(names):
    return {k: v for k, v in sorted(Counter(names).items()) if v > 1}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional new JSON report; existing files are never overwritten')
    args = parser.parse_args()
    common = {'filename', 'component', 'grid_row', 'grid_col', 'split'}
    fields, rows = read_csv(MANIFEST, common | {'platform', 'within_platform_candidate', 'cross_platform_leakage_verified', 'approved_for_training', 'approved_for_evaluation'})
    candidate_fields, candidates = read_csv(CANDIDATE, common)
    with VALIDATION.open(encoding='utf-8') as f:
        validation = json.load(f)
    errors = []
    results = {}
    for split in SPLITS:
        selected = [r for r in rows if r['split'] == split]
        reference = [r for r in candidates if r['split'] == split]
        names = [r['filename'] for r in selected]
        ref_names = [r['filename'] for r in reference]
        dup, ref_dup = duplicates(names), duplicates(ref_names)
        invalid = [n for n in names if not n or Path(n).name != n or n in ('.', '..')]
        missing = {kind: [n for n in names if n not in invalid and not (DATA / kind / n).is_file()] for kind in ('img', 'mask')}
        by_name = {r['filename']: r for r in reference}
        differing = [r['filename'] for r in selected if r['filename'] in by_name and any(r[k] != by_name[r['filename']][k] for k in common)]
        count_ref = validation['counts'].get(split)
        section = {'manifest_count': len(selected), 'candidate_count': len(reference), 'validation_record_count': count_ref,
                   'manifest_duplicate_filenames': dup, 'candidate_duplicate_filenames': ref_dup,
                   'invalid_filenames': invalid, 'missing_image': missing['img'], 'missing_mask': missing['mask'],
                   'only_in_manifest': sorted(set(names) - set(ref_names)), 'only_in_candidate': sorted(set(ref_names) - set(names)),
                   'candidate_row_mismatches': differing,
                   'outside_component_2': [r['filename'] for r in selected if r['component'] != '2'],
                   'unexpected_approval_or_candidate_flags': [r['filename'] for r in selected if any(r[k] != v for k, v in {'within_platform_candidate':'True', 'cross_platform_leakage_verified':'False', 'approved_for_training':'False', 'approved_for_evaluation':'False'}.items())]}
        results[split] = section
        if not selected or len(selected) != len(reference) or len(selected) != count_ref or any(section[k] for k in section if k not in ('manifest_count','candidate_count','validation_record_count')):
            errors.append(split + ': discrepancies found')
    overlap = sorted({r['filename'] for r in rows if r['split'] == SPLITS[0]} & {r['filename'] for r in rows if r['split'] == SPLITS[1]})
    if overlap: errors.append('train/validation filenames overlap')
    report = {'scope': 'UAV train_candidate and val_candidate only; metadata and exact file existence, no image or mask contents opened',
              'limitations': ['Filename matching does not establish geographic separation, pixel alignment, label correctness, valid masks or readability.', 'Test, buffer, unassigned and SAT files were not checked.', 'Existing validation counts are compared, not independently recomputed.'],
              'inputs_sha256': {str(p.relative_to(ROOT)): sha(p) for p in (MANIFEST, CANDIDATE, VALIDATION)},
              'manifest_fields': fields, 'candidate_fields': candidate_fields, 'splits': results,
              'train_val_filename_overlap': overlap, 'errors': errors, 'passed': not errors}
    output = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        with args.output.open('x', encoding='utf-8') as f: f.write(output)
    print(output, end='')
    return 1 if errors else 0

if __name__ == '__main__':
    raise SystemExit(main())
