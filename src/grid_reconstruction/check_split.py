"""Check existing UAV candidate split against its relative grid; no imagery needed."""
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def index(rows, name):
    out = {}
    for row in rows:
        key = row['filename']
        if key in out:
            raise ValueError(f'{name} has duplicate filename: {key}')
        out[key] = row
    return out

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise SystemExit(f'Output exists: {args.output}')
    base = args.root
    coords = index(read(base / 'data/interim/reconstructed_grids/uav_grid_coordinates_v2.csv'), 'coordinates')
    split = index(read(base / 'data/interim/splits/moxitaidi_uav_split_candidate_v1.csv'), 'split')
    if set(coords) != set(split):
        raise ValueError(f'Filename mismatch: missing={len(set(coords)-set(split))}, extra={len(set(split)-set(coords))}')
    allowed = {'train_candidate', 'val_candidate', 'test_candidate', 'buffer', 'unassigned'}
    groups = {name: [] for name in allowed}
    for filename, row in split.items():
        if row['split'] not in allowed:
            raise ValueError(f'Unknown category: {filename}: {row["split"]}')
        ref = coords[filename]
        if (row['component'], int(row['grid_row']), int(row['grid_col'])) != (ref['component'], int(ref['grid_row']), int(ref['grid_col'])):
            raise ValueError(f'Coordinate mismatch: {filename}')
        if row['split'] != 'unassigned' and row['component'] != '2':
            raise ValueError(f'Assigned outside component 2: {filename}')
        groups[row['split']].append(row)
    names = ['train_candidate', 'val_candidate', 'test_candidate']
    if any(not groups[n] for n in names):
        raise ValueError('One or more candidate splits are empty')
    pairs = []
    for i, a in enumerate(names):
        for b in names[i+1:]:
            minimum = float('inf'); violations = 0; count = 0
            for x in groups[a]:
                for y in groups[b]:
                    dx = max(abs(int(x['grid_col']) - int(y['grid_col'])) * 256 - 512, 0)
                    dy = max(abs(int(x['grid_row']) - int(y['grid_row'])) * 256 - 512, 0)
                    distance = math.hypot(dx, dy)
                    minimum = min(minimum, distance)
                    violations += distance < 512
                    count += 1
            pairs.append({'first': a, 'second': b, 'pairs': count,
                          'minimum_gap_pixels': minimum, 'violations': violations})
    passed = all(p['violations'] == 0 for p in pairs)
    report = {'status': 'candidate_within_component_check_only',
              'grid': 'uav_grid_coordinates_v2.csv', 'image_count': len(split),
              'counts': dict(Counter(row['split'] for row in split.values())),
              'tile_pixels': 512, 'grid_step_pixels': 256, 'required_gap_pixels': 512,
              'component_2_pair_checks': pairs, 'within_component_gap_passed': passed,
              'limitations': ['Other components remain unassigned; spatial separation from component 2 is unverified.',
                              'Satellite-UAV overlap, effective validity masks, and true geographic positions are unverified.',
                              'Exploratory test-area statistics were previously inspected; this is not a frozen held-out test set.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not passed: raise SystemExit('Within-component gap check failed')

if __name__ == '__main__': main()
