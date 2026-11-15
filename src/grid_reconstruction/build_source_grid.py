"""Build relative components from unique horizontal/vertical candidates."""
import argparse
import csv
import json
from collections import Counter, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASETS = {'longxi_sat': 'Longxi_SAT', 'longxi_uav': 'Longxi_UAV', 'jiuzhai_uav_05': 'Jiuzhai_UAV_0.5m'}
DIRECTIONS = {'right': (0, 1), 'down': (1, 0)}

def reconstruct(names, edges):
    graph = {name: [] for name in names}
    seen = set()
    for src, dst, direction in edges:
        key = (src, dst, direction)
        if key in seen or src not in graph or dst not in graph or src == dst or direction not in DIRECTIONS:
            raise ValueError(f'invalid or duplicate edge: {key}')
        seen.add(key)
        dr, dc = DIRECTIONS[direction]
        graph[src].append((dst, dr, dc))
        graph[dst].append((src, -dr, -dc))
    positions, sizes, conflicts = {}, [], []
    for start in names:
        if start in positions: continue
        component = len(sizes) + 1
        positions[start] = (component, 0, 0)
        queue = deque([start]); size = 0
        while queue:
            current = queue.popleft(); size += 1
            _, row, col = positions[current]
            for neighbor, dr, dc in graph[current]:
                expected = (component, row + dr, col + dc)
                if neighbor not in positions:
                    positions[neighbor] = expected; queue.append(neighbor)
                elif positions[neighbor] != expected:
                    conflicts.append(('inconsistent_position', neighbor, current, str(positions[neighbor]), str(expected)))
        sizes.append(size)
    occupied = {}
    for name, position in positions.items():
        if position in occupied:
            conflicts.append(('duplicate_position', name, occupied[position], str(position), str(position)))
        else: occupied[position] = name
    mins = {}
    for component, row, col in positions.values():
        if component not in mins: mins[component] = (row, col)
        else: mins[component] = (min(mins[component][0], row), min(mins[component][1], col))
    result = []
    for name in names:
        component, row, col = positions[name]; r0, c0 = mins[component]
        result.append({'filename': name, 'component': component, 'grid_row': row-r0, 'grid_col': col-c0, 'neighbor_count': len(graph[name])})
    return result, conflicts, sizes

def write_csv(path, fields, rows):
    with path.open('x', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator='\n'); w.writeheader(); w.writerows(rows)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True, choices=sorted(DATASETS))
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    grid = args.root / 'data/interim/reconstructed_grids'
    source = grid / f'{args.dataset}_hv_candidates_v1.csv'
    outputs = [args.output_dir / f'{args.dataset}_grid_coordinates_v1.csv', args.output_dir / f'{args.dataset}_grid_conflicts_v1.csv', args.output_dir / f'{args.dataset}_grid_summary_v1.json']
    if any(p.exists() for p in outputs): raise SystemExit('Output exists; choose a new output directory.')
    names = sorted(p.name for p in (args.root / 'data/raw/CAS' / DATASETS[args.dataset] / 'img').glob('*.tif'))
    if not names: raise SystemExit('No .tif images found.')
    with source.open(encoding='utf-8-sig', newline='') as f: rows = list(csv.DictReader(f))
    edges = [(r['source_image'], r['target_image'], r['direction']) for r in rows if r['status'] == 'unique_candidate']
    rejected = Counter(r['status'] for r in rows if r['status'] != 'unique_candidate')
    if not edges: raise SystemExit('No unique candidate edges.')
    coords, conflicts, sizes = reconstruct(names, edges)
    summary = {'dataset':args.dataset, 'image_count':len(names), 'candidate_edges':len(rows), 'used_edges':len(edges), 'skipped_by_status':dict(rejected), 'component_count':len(sizes), 'largest_components':sorted(sizes, reverse=True)[:10], 'isolated_images':sum(x==1 for x in sizes), 'conflict_count':len(conflicts), 'scope':'Relative horizontal/vertical 50% candidates only. Cross-component overlaps and diagonal links are unverified; not a data split.'}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(outputs[0], ['filename','component','grid_row','grid_col','neighbor_count'], coords)
    write_csv(outputs[1], ['type','image','other_image','existing','expected'], [dict(zip(['type','image','other_image','existing','expected'],x)) for x in conflicts])
    with outputs[2].open('x',encoding='utf-8') as f: json.dump(summary,f,ensure_ascii=False,indent=2)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    for path in outputs: print('Saved:',path)
    if conflicts: raise SystemExit('Grid has conflicts; inspect output before use.')

if __name__ == '__main__': main()
