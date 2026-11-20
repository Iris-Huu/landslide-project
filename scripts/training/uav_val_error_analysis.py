"""Fixed-checkpoint validation-only center/full and false-positive analysis."""
import csv,hashlib,json,subprocess,sys,tempfile,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
import numpy as np, torch, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from src.data.uav_data import select_samples,read_sample
from src.metrics.binary_segmentation import ConfusionMatrix
from src.metrics.region_analysis import center_mask,regional_counts,calibration_bins
from src.models.unet import SmallUNet

MANIFEST=ROOT/'data/interim/splits/moxitaidi_unified_spatial_manifest_v1.csv'; DATA=ROOT/'data/raw/CAS/Moxitaidi_UAV_0.6m'
RUNS={'bce':ROOT/'experiments/uav_train_xj_kf48q','weighted':ROOT/'experiments/uav_train_s47c7j29'}

def cm(p,t,v=None): return ConfusionMatrix().update(p,t,v).result()
def add(a,r):
    for k in ('tp','fp','fn','tn'): a[k]+=r[k]
def output(a, sample_count):
    c=ConfusionMatrix(**a, samples=sample_count); return c.result()
def spatial_stats(rows,key):
    vals=np.array([r[key] for r in rows if r[key] is not None],dtype=float)
    return {'n_defined':len(vals),'n_undefined':len(rows)-len(vals),'mean':float(vals.mean()) if len(vals) else None,'median':float(np.median(vals)) if len(vals) else None,'p25':float(np.quantile(vals,.25)) if len(vals) else None,'p75':float(np.quantile(vals,.75)) if len(vals) else None,'min':float(vals.min()) if len(vals) else None,'max':float(vals.max()) if len(vals) else None}
def main():
    torch.set_num_threads(2); device=torch.device('cuda:0'); torch.cuda.init(); torch.cuda.reset_peak_memory_stats(device)
    run=Path(tempfile.mkdtemp(prefix='uav_val_error_',dir=ROOT/'experiments')); print('run:',run,flush=True)
    subprocess.run([str(ROOT/'.venv/torch-cpu/bin/python'),'-B',str(ROOT/'scripts/training/uav_prepare_suspected.py'),str(run/'suspected_masks')],cwd=ROOT,check=True)
    samples=sorted(select_samples(MANIFEST,'val_candidate','2'),key=lambda s:s.filename)
    bins=[{'bin':i,'lower':i/10,'upper':(i+1)/10,'count':0,'probability_sum':0.,'positive_pixels':0} for i in range(10)]
    base={'created_utc':datetime.now(timezone.utc).isoformat(),'scope':'val_candidate only; prior selected checkpoints; no test/buffer/unassigned images','config':{'threshold':.5,'crop':[128,128,256,256], 'full_image':[512,512],'batch_size':4,'device':'cuda:0','valid_policy':'all_pixels_provisional','probability_bins':'10 equal bins [0,.1),...,[.9,1]'},
          'manifest_sha256':samples[0].manifest_sha256,'sample_names':[s.filename for s in samples], 'sample_sha256':hashlib.sha256(('\n'.join(s.filename for s in samples)+'\n').encode()).hexdigest(),
          'git_head':subprocess.check_output(['git','--no-optional-locks','rev-parse','HEAD'],cwd=ROOT,text=True).strip(), 'git_status':subprocess.check_output(['git','--no-optional-locks','status','--short','--untracked-files=all'],cwd=ROOT,text=True).strip(),
          'analysis_code_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),ROOT/'src/metrics/region_analysis.py',ROOT/'scripts/training/uav_prepare_suspected.py']},
          'models':{},'limitations':['same validation used for model choice; not independent test','suspected rule is unverified, may remove positive labels','overlapping tiles and pixels spatially correlated']}
    models={}
    for name,path in RUNS.items():
        cp=torch.load(path/'best.pt',map_location=device,weights_only=False); model=SmallUNet(cp['config']['base_channels']).to(device); model.load_state_dict(cp['model']); model.eval()
        record=json.loads((path/'run.json').read_text()); expected=record['epochs'][cp['epoch']-1]['validation']['confusion']
        models[name]=(model,expected)
        base['models'][name]={'checkpoint':str((path/'best.pt').relative_to(ROOT)),'checkpoint_sha256':hashlib.sha256((path/'best.pt').read_bytes()).hexdigest(),'epoch':cp['epoch'],'expected_direct_center':{k:expected[k] for k in ('tp','fp','fn','tn')}}
    pools={name:{region:dict(tp=0,fp=0,fn=0,tn=0) for region in ('direct_center','full_center','outside','full')} for name in models}
    excluded=dict(tp=0,fp=0,fn=0,tn=0); selected_omit=dict(total=0,positive=0,fp=0)
    rows=[]; heat_fp=np.zeros((16,16),dtype=np.int64); heat_bg=np.zeros((16,16),dtype=np.int64); disagreement=0; abs_prob_sum=0.
    start=time.perf_counter(); center=center_mask((512,512),256)
    direct_cache={}
    with torch.inference_mode():
        for start_idx in range(0,len(samples),4):
            group=samples[start_idx:start_idx+4]
            items=[read_sample(sample,DATA) for sample in group]
            x=torch.from_numpy(np.stack([item['image'] for item in items])).to(device)
            for name,(model,expected) in models.items():
                full_batch=torch.sigmoid(model(x))[:,0].cpu().numpy()
                direct_batch=torch.sigmoid(model(x[:,:,128:384,128:384]))[:,0].cpu().numpy()
                for sample,item,full,direct in zip(group,items,full_batch,direct_batch):
                    target=item['mask'].astype(bool); valid=item['valid_mask']
                    if target.shape!=(512,512): raise ValueError(f'{sample.filename}: unexpected dimensions')
                    target_c=target[128:384,128:384]; valid_c=valid[128:384,128:384]; pred_c=direct>=.5
                    direct_counts=cm(pred_c,target_c,valid_c); add(pools[name]['direct_center'],direct_counts)
                    reg=regional_counts(full>=.5,target,center,valid)
                    for dest,src in (('full_center','center'),('outside','outside'),('full','full')): add(pools[name][dest],reg[src])
                    if name!='weighted': continue
                    direct_cache[sample.filename]=direct
                    suspect=np.load(run/'suspected_masks'/(Path(sample.filename).stem+'.npz'))['suspected']
                    fs=full[128:384,128:384]; disagreement+=int(np.count_nonzero((fs>=.5)!=pred_c)); abs_prob_sum+=float(np.abs(fs-direct).sum(dtype=np.float64))
                    suspect_c=suspect[128:384,128:384]; keep=valid_c & ~suspect_c
                    add(excluded,cm(pred_c,target_c,keep))
                    selected_omit['total']+=int(suspect_c.sum()); selected_omit['positive']+=int((suspect_c & target_c).sum()); selected_omit['fp']+=int((suspect_c & pred_c & ~target_c).sum())
                    fp=pred_c & ~target_c & valid_c; bg=~target_c & valid_c
                    heat_fp+=fp.reshape(16,16,16,16).sum(axis=(1,3)); heat_bg+=bg.reshape(16,16,16,16).sum(axis=(1,3))
                    for dst,part in zip(bins,calibration_bins(direct,target_c,valid_c)):
                        for k in ('count','probability_sum','positive_pixels'): dst[k]+=part[k]
                    m=direct_counts['metrics']; tp,ff,fn,tn=(direct_counts[k] for k in ('tp','fp','fn','tn'))
                    rows.append({'filename':sample.filename,'tp':tp,'fp':ff,'fn':fn,'tn':tn,'iou':m['iou'],'f1':m['f1'],'precision':m['precision'],'recall':m['recall'],
                                 'fp_background_rate':ff/(ff+tn) if ff+tn else None, 'true_positive_fraction':float(target_c.mean()),'predicted_positive_fraction':float(pred_c.mean()),
                                 'suspected_pixels':int(suspect_c.sum()),'suspected_positive_pixels':int((suspect_c & target_c).sum()), 'suspected_fp':int((suspect_c & pred_c & ~target_c).sum())})
            if (start_idx+len(group))%20==0: print('processed',start_idx+len(group),'/',len(samples),flush=True)
    for name in models:
        actual=output(pools[name]['direct_center'],len(samples)); expected=base['models'][name]['expected_direct_center']; match=all(actual[k]==expected[k] for k in expected)
        base['models'][name].update({'reproduction_matches':match,'regions':{k:output(v,len(samples)) for k,v in pools[name].items()}})
        for k in ('tp','fp','fn','tn'):
            if pools[name]['full'][k]!=pools[name]['full_center'][k]+pools[name]['outside'][k]: raise AssertionError('aggregate partition mismatch')
        if not match: raise RuntimeError(name+' reproduction mismatch')
    for b in bins:
        n=b['count']; b['mean_probability']=b['probability_sum']/n if n else None; b['positive_fraction']=b['positive_pixels']/n if n else None
    chosen={}
    def pick(label,eligible,sorter):
        candidates=sorted((r for r in rows if eligible(r)),key=sorter)
        if candidates: chosen.setdefault(candidates[0]['filename'],[]).append(label)
    pick('highest_fp_count',lambda r:True,lambda r:(-r['fp'],r['filename']))
    pick('highest_fp_rate_over_background',lambda r:r['fp_background_rate'] is not None,lambda r:(-r['fp_background_rate'],r['filename']))
    pick('best_iou_positive_gt',lambda r:r['tp']+r['fn']>0,lambda r:(-r['iou'],r['filename']))
    pick('no_positive_highest_fp',lambda r:r['tp']+r['fn']==0,lambda r:(-r['fp'],r['filename']))
    weighted=models['weighted'][0]
    with torch.inference_mode():
        for name,reasons in chosen.items():
            sample=next(s for s in samples if s.filename==name); item=read_sample(sample,DATA); im=item['image'][:,128:384,128:384]; y=item['mask'][128:384,128:384].astype(bool)
            pred_prob=direct_cache[name]; pred=pred_prob>=.5
            overlay=np.transpose(im,(1,2,0)).copy(); colors=np.zeros((*y.shape,4)); colors[pred&y]=[0,1,0,.65]; colors[pred&~y]=[1,0,0,.65]; colors[~pred&y]=[0,.5,1,.75]
            fig,ax=plt.subplots(1,5,figsize=(19,4),constrained_layout=True); ax[0].imshow(overlay); ax[0].set_title('RGB')
            ax[1].imshow(y,cmap='gray',vmin=0,vmax=1); ax[1].set_title('Label')
            img=ax[2].imshow(pred_prob,cmap='viridis',vmin=0,vmax=1); ax[2].set_title('Probability'); fig.colorbar(img,ax=ax[2],fraction=.046)
            ax[3].imshow(pred,cmap='gray',vmin=0,vmax=1); ax[3].set_title('Prediction >=0.5')
            ax[4].imshow(overlay); ax[4].imshow(colors); ax[4].set_title('TP green / FP red / FN blue')
            for a in ax: a.axis('off')
            fig.suptitle(f'{name} | weighted best epoch 20 | 256 center (128,128) | {", ".join(reasons)}')
            path=run/(Path(name).stem+'_diagnosis.png'); fig.savefig(path,dpi=130); plt.close(fig)
            with Image.open(path) as f: f.verify()
    np.savez_compressed(run/'fp_spatial_grid.npz',fp=heat_fp,background=heat_bg)
    with np.errstate(divide='ignore',invalid='ignore'): rate=np.divide(heat_fp,heat_bg,out=np.zeros_like(heat_fp,dtype=float),where=heat_bg>0)
    fig,ax=plt.subplots(figsize=(6,5)); im=ax.imshow(rate,origin='upper',cmap='magma',vmin=0,vmax=1); ax.set_title('FP / background by 16x16 crop grid'); fig.colorbar(im,ax=ax); fig.savefig(run/'fp_spatial_rate.png',dpi=140,bbox_inches='tight'); plt.close(fig)
    with (run/'per_image.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (run/'per_image.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    base.update({'per_image_distributions':{k:spatial_stats(rows,k) for k in ('fp','fp_background_rate','iou','precision','recall','true_positive_fraction','predicted_positive_fraction')},
                 'sensitivity_excluding_unverified_suspected':{'counts':output(excluded,len(samples)),'excluded':selected_omit,'rule':'edge-connected pure black or white on full image, cropped to center; previous audit counts matched'},
                 'probability_bins':bins,'selected_examples':chosen,'center_context_comparison':{'disagreement_pixels':disagreement,'total_pixels':len(samples)*256*256,'mean_absolute_probability_difference':abs_prob_sum/(len(samples)*256*256)},
                 'spatial_fp_grid':{'fp':heat_fp.tolist(),'background':heat_bg.tolist()},'seconds':time.perf_counter()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(device)})
    (run/'summary.json').write_text(json.dumps(base,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print('completed',run,flush=True)
if __name__=='__main__': main()
