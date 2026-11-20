"""Synthetic-only SegFormer-B2 forward/backward check; never reads real data."""
import json,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
import segmentation_models_pytorch as smp, torch
from src.training.segmentation_steps import train_step

def main():
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(2);torch.manual_seed(17);torch.cuda.manual_seed_all(17);torch.cuda.init();torch.cuda.reset_peak_memory_stats()
    run=Path(tempfile.mkdtemp(prefix='segformer_b2_synthetic_',dir=ROOT/'experiments')); file=run/'run.json'
    info={'scope':'synthetic only, no imagery','architecture':'SMP Segformer mit_b2','smp':smp.__version__,'torch':torch.__version__,'device':torch.cuda.get_device_name(0),'weights':'none; random initialization','seed':17,'status':'started'}
    file.write_text(json.dumps(info,indent=2)+'\n');print('run:',run,flush=True)
    try:
        model=smp.Segformer(encoder_name='mit_b2',encoder_weights=None,in_channels=3,classes=1,activation=None).to('cuda:0');opt=torch.optim.Adam(model.parameters(),lr=.0001)
        x=torch.rand(2,3,128,128,device='cuda:0'); y=torch.randint(0,2,(2,1,128,128),device='cuda:0').float(); valid=torch.ones_like(y,dtype=torch.bool)
        with torch.no_grad(): model.eval(); z=model(x); shape=list(z.shape)
        if shape!=[2,1,128,128]: raise ValueError(f'unexpected output {shape}')
        before=next(model.parameters()).detach().clone(); t=time.perf_counter(); loss=train_step(model,opt,x,y,valid,loss_kind='dice_bce');torch.cuda.synchronize()
        if not torch.isfinite(torch.tensor(loss)) or torch.equal(before,next(model.parameters())): raise RuntimeError('loss/update failed')
        info.update({'status':'completed','parameter_count':sum(p.numel() for p in model.parameters()),'input_shape':list(x.shape),'output_shape':shape,'loss_kind':'Dice+BCE','loss':loss,'parameter_changed':True,'seconds':time.perf_counter()-t,'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
    except Exception as e:
        info.update({'status':'failed','error':f'{type(e).__name__}: {e}'});raise
    finally: file.write_text(json.dumps(info,indent=2,allow_nan=False)+'\n')
    print(json.dumps(info,indent=2),flush=True)
if __name__=='__main__': main()
