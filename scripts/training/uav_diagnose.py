"""Candidate train/val center-crop diagnostics for trusted checkpoints; no test access."""
import csv, hashlib, json, subprocess, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.models.unet import SmallUNet
from src.training.exploratory import CandidateDataset
from src.metrics.binary_segmentation import ConfusionMatrix

MANIFEST=ROOT/'data/interim/splits/moxitaidi_unified_spatial_manifest_v1.csv'; DATA=ROOT/'data/raw/CAS/Moxitaidi_UAV_0.6m'
ORIG=ROOT/'experiments/uav_train_0ev4mlp7'; AUDIT=ROOT/'reports/tables/moxitaidi_basics_per_image.csv'
THRESHOLDS=[i/100 for i in range(5,100,5)]

def stats(a):
    if len(a)==0: return None
    return {'mean':float(np.mean(a)), 'min':float(np.min(a)), 'p05':float(np.quantile(a,.05)), 'p25':float(np.quantile(a,.25)), 'median':float(np.median(a)), 'p75':float(np.quantile(a,.75)), 'p95':float(np.quantile(a,.95)), 'max':float(np.max(a))}
def main():
    torch.set_num_threads(2); d=torch.device('cuda:0')
    run=Path(tempfile.mkdtemp(prefix='uav_diagnostic_',dir=ROOT/'experiments')); print('run:',run,flush=True)
    rec={'created_utc':datetime.now(timezone.utc).isoformat(),'scope':'train_candidate and val_candidate center crops only', 'threshold_grid':THRESHOLDS,'threshold_note':'validation-set tuning; optimistic, not independent test','crop':'256x256 center (top=128,left=128)', 'valid_policy':'all_pixels_provisional','data':{},'checkpoints':{}}
    audit={}
    with AUDIT.open(newline='',encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            if row['subset']=='Moxitaidi_UAV_0.6m': audit[row['filename']]=float(row['landslide_percent'])
    datasets={}
    for split in ('train_candidate','val_candidate'):
        ds=CandidateDataset(MANIFEST,DATA,split,crop=256); datasets[split]=ds
        counts=[]
        for i in range(len(ds)):
            _,y,v=ds[i]; counts.append(int(y.sum()))
        arr=np.array(counts); full=[audit[s.filename] for s in ds.samples]
        rec['data'][split]={'count':len(ds),'positive_pixels':int(arr.sum()),'total_pixels':int(len(ds)*256*256), 'positive_percent':float(arr.sum()/(len(ds)*256*256)*100), 'zero_positive_crops':int((arr==0).sum()),'per_crop_positive_percent':stats(arr/(256*256)*100), 'full_image_historical_mean_percent':float(np.mean(full)),'full_image_historical_median_percent':float(np.median(full)),'full_image_historical_rows':len(full)}
        rec['data'][split]['per_sample']=[{'filename':s.filename,'positive_pixels':int(n),'full_image_historical_percent':audit[s.filename]} for s,n in zip(ds.samples,counts)]
    val=datasets['val_candidate']; loader=DataLoader(val,batch_size=4,shuffle=False,num_workers=0)
    for ck in ('best.pt','last.pt'):
        cp=torch.load(ORIG/ck,map_location=d,weights_only=False); model=SmallUNet(cp['config']['base_channels']).to(d); model.load_state_dict(cp['model']); model.eval()
        probs=[]; labels=[]
        with torch.no_grad():
            for x,y,v in loader:
                pr=torch.sigmoid(model(x.to(d))).cpu().numpy(); probs.append(pr[v.numpy()]); labels.append(y.numpy()[v.numpy()])
        prob=np.concatenate(probs); target=np.concatenate(labels).astype(bool); del probs,labels
        thresholds={}
        for threshold in THRESHOLDS:
            meter=ConfusionMatrix().update(prob>=threshold,target)
            thresholds[f'{threshold:.2f}']=meter.result()
        key=max(thresholds,key=lambda k:(thresholds[k]['metrics']['f1'] if thresholds[k]['metrics']['f1'] is not None else -1,float(k)))
        rec['checkpoints'][ck]={'epoch':cp['epoch'],'sha256':hashlib.sha256((ORIG/ck).read_bytes()).hexdigest(),'probability_all':stats(prob),'probability_positive':stats(prob[target]),'probability_background':stats(prob[~target]),'actual_positive_ratio':float(target.mean()),'predicted_positive_ratio_at_0_5':float((prob>=.5).mean()),'thresholds':thresholds,'best_validation_f1_threshold':key}
        old=json.load(open(ORIG/'run.json'))['epochs'][cp['epoch']-1]['validation']['confusion']
        now=thresholds['0.50']; rec['checkpoints'][ck]['reproduces_recorded_confusion']=all(now[k]==old[k] for k in ('tp','fp','fn','tn'))
        print(ck,'epoch',cp['epoch'],'reproduces',rec['checkpoints'][ck]['reproduces_recorded_confusion'],'best_threshold',key,flush=True)
        if ck=='last.pt':
            for split,ds in datasets.items():
                selection=[i for i,r in enumerate(rec['data'][split]['per_sample']) if r['positive_pixels']>0][:2]
                for idx in selection:
                    x,y,v=ds[idx]
                    with torch.no_grad(): pr=torch.sigmoid(model(x.unsqueeze(0).to(d)))[0,0].cpu().numpy()
                    fig,axes=plt.subplots(1,4,figsize=(15,4),constrained_layout=True)
                    axes[0].imshow(x.permute(1,2,0)); axes[0].set_title('RGB center crop')
                    axes[1].imshow(y[0],cmap='gray',vmin=0,vmax=1); axes[1].set_title('Ground truth 0/1')
                    im=axes[2].imshow(pr,cmap='viridis',vmin=0,vmax=1); axes[2].set_title('Probability [0,1]'); fig.colorbar(im,ax=axes[2],fraction=.046)
                    axes[3].imshow(pr>=float(key),cmap='gray',vmin=0,vmax=1); axes[3].set_title(f'Prediction threshold {key}*')
                    for ax in axes: ax.axis('off')
                    name=ds.samples[idx].filename; fig.suptitle(f'{split} {name} | last.pt epoch 3 | center crop (128,128) | *val-selected threshold')
                    fig.savefig(run/f'{split}_{Path(name).stem}.png',dpi=120); plt.close(fig)
    rec['manifest_sha256']=datasets['train_candidate'].samples[0].manifest_sha256
    (run/'diagnosis.json').write_text(json.dumps(rec,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print('summary:',run/'diagnosis.json',flush=True)
    if not all(v['reproduces_recorded_confusion'] for v in rec['checkpoints'].values()): raise RuntimeError('baseline reproduction mismatch')
if __name__=='__main__': main()
