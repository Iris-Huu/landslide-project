"""Preview selected cross-component diagonal candidates; no automatic merging."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
import warnings

warnings.filterwarnings('ignore',category=NotGeoreferencedWarning)
ROOT=Path(__file__).resolve().parents[2]
SPECS=[('longxi_sat','Longxi_SAT','5','10',3),('longxi_uav','Longxi_UAV','1','3',1),('jiuzhai_uav_05','Jiuzhai_UAV_0.5m','10','15',2)]

def load_image(path):
    with rasterio.open(path) as src: data=src.read()
    if data.shape != (3,512,512): raise ValueError(f'unexpected image shape {path}: {data.shape}')
    return np.moveaxis(data,0,-1)

def corners(source,target,direction):
    if direction=='down_right':
        return source[256:,256:],target[:256,:256],(256,256),(0,0)
    if direction=='down_left':
        return source[256:,:256],target[:256,256:],(0,256),(256,0)
    raise ValueError(f'unsupported direction: {direction}')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--figure',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    a=p.parse_args()
    if a.figure.exists() or a.manifest.exists():raise SystemExit('Output exists; select new paths')
    selected=[]
    for dataset,folder,source_group,target_group,expected in SPECS:
        path=a.root/'data/interim/reconstructed_grids'/f'{dataset}_diagonal_diagnostics_v1.csv'
        with path.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
        matches=[r for r in rows if r['candidate_status']=='unique_candidate' and r['relation']=='cross_group' and r['source_component']==source_group and r['target_component']==target_group]
        if len(matches)!=expected:raise ValueError(f'{dataset} group {source_group}->{target_group}: expected {expected} rows; got {len(matches)}')
        shifts={(r['target_shift_row'],r['target_shift_col']) for r in matches}
        if len(shifts)!=1:raise ValueError(f'{dataset}: group proposals disagree: {shifts}')
        for r in matches:selected.append((dataset,folder,r))
    fig,axes=plt.subplots(len(selected),3,figsize=(12,3.5*len(selected)),squeeze=False)
    manifest=[]
    try:
        with rasterio.Env(GDAL_CACHEMAX=32*1024*1024):
            for i,(dataset,folder,row) in enumerate(selected):
                names=[row['source_image'],row['target_image']]
                if any(Path(n).name!=n for n in names):raise ValueError('Unexpected filename')
                source,target=(load_image(a.root/'data/raw/CAS'/folder/'img'/n) for n in names)
                left,right,srcxy,dstxy=corners(source,target,row['direction'])
                if not np.array_equal(left,right): raise ValueError(f'overlap differs: {names}')
                axes[i,0].imshow(source);axes[i,0].add_patch(Rectangle(srcxy,256,256,fill=False,edgecolor='yellow',linewidth=2));axes[i,0].set_title(f'{dataset} source: {names[0]}',fontsize=8)
                axes[i,1].imshow(target);axes[i,1].add_patch(Rectangle(dstxy,256,256,fill=False,edgecolor='yellow',linewidth=2));axes[i,1].set_title(f'target: {names[1]}',fontsize=8)
                axes[i,2].imshow(left);axes[i,2].set_title(f'Exact overlap; {row["direction"]}\nblack/white: {float(row["pure_black_white_percent"]):.2f}%',fontsize=8)
                for ax in axes[i]:ax.axis('off')
                manifest.append({'dataset':dataset,'source':names[0],'target':names[1],'source_component':row['source_component'],'target_component':row['target_component'],'direction':row['direction'],'target_shift_row':int(row['target_shift_row']),'target_shift_col':int(row['target_shift_col']),'pure_black_white_percent':float(row['pure_black_white_percent']),'pixel_equal':True})
        a.figure.parent.mkdir(parents=True,exist_ok=True);a.manifest.parent.mkdir(parents=True,exist_ok=True)
        fig.suptitle('Cross-component diagonal candidates: manual review required',fontsize=13);fig.tight_layout();fig.savefig(a.figure,dpi=120,bbox_inches='tight')
    finally:plt.close(fig)
    with a.manifest.open('x',encoding='utf-8') as f:json.dump({'status':'unreviewed_candidates','pairs':manifest,'note':'Matching overlap and a visually structured region do not alone prove global component placement.'},f,ensure_ascii=False,indent=2)
    print('Selected pairs:',len(manifest));print('Figure:',a.figure);print('Manifest:',a.manifest);print('Manual review required; no coordinates were changed.')

if __name__=='__main__':main()
