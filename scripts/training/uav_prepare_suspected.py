"""Recompute existing edge-connected candidate rule on val candidates only."""
import csv, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
import numpy as np
from src.data.uav_data import select_samples,read_sample
from src.data_audit.audit_nodata_candidate import find_candidate

out=Path(sys.argv[1]); out.mkdir(parents=True,exist_ok=False)
manifest=ROOT/'data/interim/splits/moxitaidi_unified_spatial_manifest_v1.csv'; data=ROOT/'data/raw/CAS/Moxitaidi_UAV_0.6m'
with (ROOT/'reports/tables/moxitaidi_nodata_audit_v1.csv').open(newline='',encoding='utf-8-sig') as f:
    audit={r['filename']:r for r in csv.DictReader(f) if r['subset']=='Moxitaidi_UAV_0.6m'}
rows=[]
for sample in sorted(select_samples(manifest,'val_candidate','2'),key=lambda s:s.filename):
    item=read_sample(sample,data); raw=np.rint(item['image']*255).astype(np.uint8); mask=find_candidate(raw); y=item['mask']
    old=audit[sample.filename]
    if int(mask.sum())!=int(old['candidate_pixels']) or int((mask & (y==1)).sum())!=int(old['overlap_pixels']):
        raise ValueError(f'{sample.filename}: differs from prior audit')
    np.savez_compressed(out/(Path(sample.filename).stem+'.npz'),suspected=mask)
    rows.append({'filename':sample.filename,'suspected_pixels':int(mask.sum()),'overlap_positive':int((mask & (y==1)).sum())})
(out/'index.json').write_text(json.dumps({'rule':'existing find_candidate, full image, prior audit counts verified','samples':rows},indent=2)+'\n')
print('prepared',len(rows),'candidate masks; audit counts match')
