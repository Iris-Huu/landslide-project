"""Verify all grid-implied within-component overlaps and save per-pair results."""
import argparse
import csv
import os
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import rasterio
from validate_grid_overlaps import read_coordinates, read_image, get_overlap

DIRECTIONS = [('right',0,1),('down',1,0),('down_right',1,1),('down_left',1,-1)]
FIELDS = ['component','source_image','target_image','direction','overlap_pixels','different_pixels','status','error']

def iter_pairs(positions):
    for (component,row,col), source in positions.items():
        for direction,dr,dc in DIRECTIONS:
            target = positions.get((component,row+dr,col+dc))
            if target is not None:
                yield component,source,target,direction,dr,dc

def compare(source,target,dr,dc):
    left,right = get_overlap(source,target,dr,dc)
    if left.shape != right.shape or left.size == 0:
        raise ValueError('empty or mismatched overlap')
    difference = np.any(left != right, axis=0)
    return int(difference.size), int(difference.sum())

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images',type=Path,required=True)
    p.add_argument('--coordinates',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if not args.images.is_dir(): raise SystemExit('Image directory does not exist')
    partial=args.output.with_name(args.output.name+'.partial')
    if args.output.exists() or partial.exists(): raise SystemExit('Output or partial file already exists; inspect before retrying')
    positions=read_coordinates(args.coordinates)
    pairs=list(iter_pairs(positions))
    if not pairs: raise SystemExit('No within-component overlap pairs; cannot validate')
    counts=Counter(); args.output.parent.mkdir(parents=True,exist_ok=True)
    @lru_cache(maxsize=4)
    def load(name): return read_image(args.images,name)
    with rasterio.Env(GDAL_CACHEMAX=32*1024*1024):
        with partial.open('x',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=FIELDS,lineterminator='\n');writer.writeheader()
            for i,(component,src,dst,direction,dr,dc) in enumerate(pairs,1):
                row={'component':component,'source_image':src,'target_image':dst,'direction':direction,'overlap_pixels':'','different_pixels':'','status':'','error':''}
                try:
                    pixels,different=compare(load(src),load(dst),dr,dc)
                    row.update(overlap_pixels=pixels,different_pixels=different,status='identical' if different == 0 else 'different')
                except Exception as error:
                    row.update(status='read_error',error=f'{type(error).__name__}: {error}')
                counts[row['status']]+=1;writer.writerow(row)
                if i%500==0 or i==len(pairs): print(f'  processed {i}/{len(pairs)}',flush=True)
    # Atomic hard-link creation protects any concurrently created final output.
    os.link(partial,args.output)
    partial.unlink()
    print('Images:',len(positions),'Pairs:',len(pairs))
    print('Identical:',counts['identical'],'Different:',counts['different'],'Read errors:',counts['read_error'])
    print('Results:',args.output)
    print('Only within components; different components and unmodelled shifts are not checked.')
    if counts['different'] or counts['read_error']: raise SystemExit('Overlap validation found discrepancies; inspect CSV')

if __name__=='__main__': main()
