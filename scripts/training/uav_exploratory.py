"""Run scoped GPU-only UAV exploratory checks or training; never read test candidates."""
import argparse, hashlib, json, random, subprocess, sys, tempfile, time
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader
from src.models.unet import SmallUNet
from src.training.segmentation_steps import train_step, masked_bce
from src.training.exploratory import CandidateDataset, evaluate, save_checkpoint, load_checkpoint

MANIFEST=ROOT/'data/interim/splits/moxitaidi_unified_spatial_manifest_v1.csv'
DATA=ROOT/'data/raw/CAS/Moxitaidi_UAV_0.6m'

def git(*args): return subprocess.check_output(['git','--no-optional-locks',*args],cwd=ROOT,text=True).strip()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('smoke','overfit','benchmark','train'),required=True)
    p.add_argument('--epochs',type=int,default=3); p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--crop',type=int,default=256); p.add_argument('--seed',type=int,default=17)
    p.add_argument('--base-channels',type=int,default=8); p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--pos-weight',type=float,default=None,help='Fixed positive BCE weight derived only from training data')
    p.add_argument('--resume',type=Path,help='Trusted local checkpoint only; starts a new run directory')
    a=p.parse_args()
    if a.batch_size<1 or a.epochs<1 or a.crop<4 or a.crop%4: p.error('invalid configuration')
    if a.pos_weight is not None and (not np.isfinite(a.pos_weight) or a.pos_weight<=0): p.error('pos-weight must be positive and finite')
    if a.resume and a.mode!='train': p.error('resume only for training')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(2); torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    device=torch.device('cuda:0'); torch.cuda.init(); torch.cuda.reset_peak_memory_stats(device)
    run=Path(tempfile.mkdtemp(prefix='uav_'+a.mode+'_',dir=ROOT/'experiments'))
    runfile=run/'run.json'; config=vars(a).copy(); config['resume']=str(a.resume) if a.resume else None
    config.update({'device':'cuda:0','valid_policy':'all_pixels_provisional','threshold':0.5,'selection':'UAV component 2 candidates only','crop_strategy':'center crop same for image/mask/valid','training_order':'per-epoch seed + epoch random permutation', 'normalization':'uint8 / 255', 'optimizer':'Adam','loss':'masked BCEWithLogits','validation_loss':'unweighted masked BCEWithLogits for cross-run comparison','workers':0})
    record={'status':'started','started_utc':datetime.now(timezone.utc).isoformat(),'config':config,'limitations':['exploratory candidate split only; not formal E1 or independent test','all pixels provisional; valid area unverified','test candidates are never loaded'],
            'environment':{'python':sys.version,'torch':torch.__version__,'torch_cuda_build':torch.version.cuda,'rasterio':rasterio.__version__,'numpy':np.__version__,'gpu':torch.cuda.get_device_name(0),'gpu_total_bytes':torch.cuda.get_device_properties(0).total_memory,'driver':subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip(),'threads':torch.get_num_threads()},
            'git_head':git('rev-parse','HEAD'),'git_status':git('status','--short','--untracked-files=all'),
            'source_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (Path(__file__),ROOT/'src/training/exploratory.py',ROOT/'src/models/unet.py',ROOT/'src/training/segmentation_steps.py',ROOT/'src/data/uav_data.py',ROOT/'src/metrics/binary_segmentation.py')},'epochs':[]}
    def save(): runfile.write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    save(); print('run:',run,flush=True)
    try:
        train=CandidateDataset(MANIFEST,DATA,'train_candidate',limit=2 if a.mode=='smoke' else 16 if a.mode=='benchmark' else None,crop=a.crop)
        if a.mode=='overfit':
            selected=[]
            for index, sample in enumerate(train.samples):
                if train[index][1].sum().item()>0:
                    selected.append(sample)
                if len(selected)==2: break
            if len(selected)<2: raise ValueError('fewer than two positive train crops')
            train.samples=selected
            record['overfit_selection']='first two lexically sorted training center crops containing positive pixels'
        val=CandidateDataset(MANIFEST,DATA,'val_candidate',limit=4 if a.mode=='benchmark' else None,crop=a.crop) if a.mode in ('benchmark','train') else None
        record['manifest_sha256']=train.samples[0].manifest_sha256
        record['train_samples']=[x.filename for x in train.samples]; record['val_samples']=[x.filename for x in val.samples] if val else []
        model=SmallUNet(a.base_channels).to(device); opt=torch.optim.Adam(model.parameters(),lr=a.lr)
        record['model_parameter_count']=sum(x.numel() for x in model.parameters()); start_epoch=0; best=-1.
        if a.resume:
            s=load_checkpoint(a.resume,model,opt,device)
            saved_config=dict(s['config']); saved_config.setdefault('pos_weight',None)
            if saved_config!= {k:config[k] for k in ('base_channels','lr','crop','seed','batch_size','pos_weight')} : raise ValueError('resume configuration mismatch')
            start_epoch=s['epoch']; best=s['best']
        cp_config={k:config[k] for k in ('base_channels','lr','crop','seed','batch_size','pos_weight')}
        save()
        if a.mode in ('smoke','overfit'):
            batch=next(iter(DataLoader(train,batch_size=2,shuffle=False,num_workers=0))); x,y,v=(z.to(device) for z in batch)
            with torch.no_grad(): initial=float(masked_bce(model(x),y,v))
            before=[q.detach().clone() for q in model.parameters()]
            steps=2 if a.mode=='smoke' else 30
            t=time.perf_counter(); losses=[]
            for i in range(steps): losses.append(train_step(model,opt,x,y,v))
            torch.cuda.synchronize()
            with torch.no_grad(): final=float(masked_bce(model(x),y,v))
            record.update({'positive_pixels_per_crop':[int(z.sum()) for z in y], 'steps':steps,'seconds':time.perf_counter()-t,'initial_loss':initial,'final_loss':final,'losses':losses,'parameters_changed':any(not torch.equal(q,b) for q,b in zip(model.parameters(),before))})
            if not record['parameters_changed'] or (a.mode=='overfit' and not final<initial): raise ValueError('smoke/overfit check failed')
        else:
            for epoch in range(start_epoch,a.epochs if a.mode=='train' else 1):
                generator=torch.Generator().manual_seed(a.seed+epoch)
                loader=DataLoader(train,batch_size=a.batch_size,shuffle=True,num_workers=0,generator=generator)
                t=time.perf_counter(); total=0.; count=0
                for x,y,v in loader:
                    x,y,v=x.to(device),y.to(device),v.to(device); n=int(v.sum().item())
                    loss=train_step(model,opt,x,y,v,pos_weight=a.pos_weight); total+=loss*n; count+=n
                torch.cuda.synchronize(); train_seconds=time.perf_counter()-t
                vstart=time.perf_counter(); metrics=evaluate(model,DataLoader(val,batch_size=a.batch_size,shuffle=False,num_workers=0),device)
                torch.cuda.synchronize(); val_seconds=time.perf_counter()-vstart
                epoch_result={'epoch':epoch+1,'train_loss':total/count,'train_seconds':train_seconds,'val_seconds':val_seconds,'validation':metrics,'peak_allocated_bytes':torch.cuda.max_memory_allocated(device)}
                record['epochs'].append(epoch_result)
                score=metrics['confusion']['metrics']['iou']
                if a.mode=='train':
                    if score is not None and score>best:
                        best=score; save_checkpoint(run/'best.pt',model,opt,epoch+1,cp_config,best); record['best_epoch']=epoch+1
                    save_checkpoint(run/'last.pt',model,opt,epoch+1,cp_config,best)
                save(); print(json.dumps(epoch_result,ensure_ascii=False),flush=True)
        record['peak_allocated_bytes']=torch.cuda.max_memory_allocated(device); record['status']='completed'
    except Exception as exc:
        record['status']='failed'; record['error']=f'{type(exc).__name__}: {exc}'; raise
    finally:
        record['finished_utc']=datetime.now(timezone.utc).isoformat(); save()
    print(json.dumps({'run':str(run),'status':record['status'],'peak_allocated_bytes':record['peak_allocated_bytes']},ensure_ascii=False),flush=True)
if __name__=='__main__': main()
