"""Safely reuse diagonal matching and compare candidates with the source v1 grid."""
import argparse
from collections import Counter
from pathlib import Path
import rasterio
import find_diagonal_matches_all as matcher
import check_diagonal_consistency_all as diagnostic

ROOT = Path(__file__).resolve().parents[2]
DATASETS = {'longxi_sat':'Longxi_SAT','longxi_uav':'Longxi_UAV','jiuzhai_uav_05':'Jiuzhai_UAV_0.5m'}
LIMIT = 50_000

def estimate(images):
    target={'down_right':Counter(),'down_left':Counter()}
    source={'down_right':Counter(),'down_left':Counter()}
    self_count=Counter()
    for index,path in enumerate(images,1):
        image=matcher.read_image(path)
        tl=matcher.image_hash(image[:,:256,:256]);tr=matcher.image_hash(image[:,:256,256:])
        bl=matcher.image_hash(image[:,256:,:256]);br=matcher.image_hash(image[:,256:,256:])
        target['down_right'][tl]+=1;source['down_right'][br]+=1;self_count['down_right']+=int(br==tl)
        target['down_left'][tr]+=1;source['down_left'][bl]+=1;self_count['down_left']+=int(bl==tr)
        if index%200==0 or index==len(images): print(f'Precheck {index}/{len(images)}',flush=True)
    return {d:sum(n*target[d].get(k,0) for k,n in source[d].items())-self_count[d] for d in target}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=sorted(DATASETS),required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--root',type=Path,default=ROOT)
    a=p.parse_args(); grid=a.root/'data/interim/reconstructed_grids'; images_dir=a.root/'data/raw/CAS'/DATASETS[a.dataset]/'img'
    coord=grid/f'{a.dataset}_grid_coordinates_v1.csv'
    out=a.output_dir/f'{a.dataset}_diagonal_candidates_v1.csv'
    diag=a.output_dir/f'{a.dataset}_diagonal_diagnostics_v1.csv'
    if out.exists() or diag.exists(): raise SystemExit('Output exists; inspect before rerun')
    if not coord.is_file(): raise SystemExit(f'Grid missing: {coord}')
    images=sorted(images_dir.glob('*.tif'))
    if not images: raise SystemExit(f'Images missing: {images_dir}')
    with rasterio.Env(GDAL_CACHEMAX=32*1024*1024):
        counts=estimate(images);print('Possible fingerprint pairs:',counts,flush=True)
        if sum(counts.values())>LIMIT: raise SystemExit(f'Candidate estimate exceeds {LIMIT}; no result written')
        matcher.IMAGE_DIR=images_dir;matcher.OUTPUT_PATH=out;matcher.main()
    diagnostic.COORD_PATH=coord;diagnostic.DIAG_PATH=out;diagnostic.OUTPUT_PATH=diag;diagnostic.main()
    print('Candidate and diagnostic outputs:',out,diag)
    print('Unique cross-component links are clues, not confirmed merges.')

if __name__=='__main__': main()
