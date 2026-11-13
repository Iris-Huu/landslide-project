#!/usr/bin/env python3
"""Rebuild Moxitaidi relative grids using recorded candidate edges only.

v1: unique horizontal/vertical edges. v2: v1 plus explicitly reviewed diagonal
edges. Coordinates are relative to each component, not geographic coordinates.
"""
import argparse
import csv
from collections import deque
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
GRID = PROJECT / 'data/interim/reconstructed_grids'
SELECTED = {
    'uav': {
        ('luding_UAV0150.tif','luding_UAV0154.tif','down_right'),
        ('luding_UAV0802.tif','luding_UAV0820.tif','down_right'),
    },
    'satellite': {
        ('Moxitaidi_Satellite245.tif','Moxitaidi_Satellite251.tif','down_right'),
        ('Moxitaidi_Satellite263.tif','Moxitaidi_Satellite264.tif','down_right'),
        ('Moxitaidi_Satellite548.tif','Moxitaidi_Satellite551.tif','down_right'),
        ('Moxitaidi_Satellite557.tif','Moxitaidi_Satellite563.tif','down_right'),
        ('Moxitaidi_Satellite582.tif','Moxitaidi_Satellite593.tif','down_right'),
    },
}
OFFSETS={'right':(0,1),'down':(1,0),'down_right':(1,1),'down_left':(1,-1)}


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def write(path, fields, rows):
    with path.open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['uav','satellite'],required=True)
    p.add_argument('--version',type=int,choices=[1,2],required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    prefix=args.dataset
    coords_path=args.output_dir/f'{prefix}_grid_coordinates_v{args.version}.csv'
    conflicts_path=args.output_dir/f'{prefix}_grid_conflicts_v{args.version}.csv'
    for path in (coords_path,conflicts_path):
        if path.exists(): raise SystemExit(f'Output exists: {path}')
    subset='Moxitaidi_UAV_0.6m' if prefix=='uav' else 'Moxitaidi_SAT'
    image_dir=PROJECT/'data/raw/CAS'/subset/'img'
    names=sorted(path.name for path in image_dir.glob('*.tif'))
    if not names: raise SystemExit(f'No TIFF images: {image_dir}')
    graph={name:[] for name in names}

    def edge(source,target,direction):
        if source not in graph or target not in graph:
            raise ValueError(f'Unknown image: {source}, {target}')
        if source==target or direction not in OFFSETS:
            raise ValueError(f'Invalid edge: {source}, {target}, {direction}')
        dr,dc=OFFSETS[direction]
        graph[source].append((target,dr,dc));graph[target].append((source,-dr,-dc))

    skipped=0;used=0
    if prefix=='uav':
        for row in read(GRID/'uav_horizontal_candidates.csv'):
            if row['status']!='unique_candidate': skipped+=1;continue
            edge(row['left_image'],row['right_image'],'right');used+=1
        for row in read(GRID/'uav_vertical_candidates.csv'):
            if row['status']!='unique_candidate': skipped+=1;continue
            edge(row['top_image'],row['bottom_image'],'down');used+=1
    else:
        for row in read(GRID/'satellite_horizontal_vertical_candidates.csv'):
            if row['status']!='unique_candidate': skipped+=1;continue
            if row['direction'] not in ('right','down'): raise ValueError('Unexpected direction')
            edge(row['source_image'],row['target_image'],row['direction']);used+=1

    found=set()
    if args.version==2:
        for row in read(GRID/f'{prefix}_diagonal_candidates.csv'):
            key=(row['source_image'],row['target_image'],row['direction'])
            if key not in SELECTED[prefix]: continue
            if key in found or row['status']!='unique_candidate':
                raise ValueError(f'Non-unique or repeated selected diagonal: {key}')
            edge(*key);found.add(key)
        if found!=SELECTED[prefix]: raise ValueError(f'Missing reviewed diagonals: {SELECTED[prefix]-found}')

    positions={}; conflicts=[];sizes=[]
    for start in names:
        if start in positions: continue
        component=len(sizes)+1
        positions[start]=(component,0,0);q=deque([start]);size=0
        while q:
            current=q.popleft();size+=1
            _,r,c=positions[current]
            for neighbor,dr,dc in graph[current]:
                expected=(component,r+dr,c+dc)
                if neighbor not in positions:
                    positions[neighbor]=expected;q.append(neighbor)
                elif positions[neighbor]!=expected:
                    conflicts.append({'type':'inconsistent_position','component':component,'image':neighbor,
                                      'other_image':current,'detail':f'existing={positions[neighbor]}, expected={expected}'})
        sizes.append(size)

    occupied={}
    for name,pos in positions.items():
        if pos in occupied:
            conflicts.append({'type':'duplicate_position','component':pos[0],'image':name,
                              'other_image':occupied[pos],'detail':f'position={pos}'})
        else: occupied[pos]=name
    mins={}
    for comp,r,c in positions.values():
        if comp not in mins: mins[comp]=[r,c]
        else: mins[comp]=[min(mins[comp][0],r),min(mins[comp][1],c)]
    rows=[]
    for name in names:
        comp,r,c=positions[name];mr,mc=mins[comp]
        rows.append({'filename':name,'component':comp,'grid_row':r-mr,'grid_col':c-mc,'neighbor_count':len(graph[name])})
    args.output_dir.mkdir(parents=True,exist_ok=True)
    write(coords_path,['filename','component','grid_row','grid_col','neighbor_count'],rows)
    write(conflicts_path,['type','component','image','other_image','detail'],conflicts)
    print(f'{prefix} v{args.version}: {len(names)} images, {len(sizes)} components, {len(conflicts)} conflict records')
    print(f'Used {used} horizontal/vertical edges, skipped {skipped} ambiguous, selected {len(found)} diagonals')
    print(f'Coordinates: {coords_path}\nConflicts: {conflicts_path}')
    print('Different components may still overlap; coordinates are not geographic.')
    if conflicts: raise SystemExit(1)


if __name__=='__main__': main()
