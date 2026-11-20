"""Exploratory candidate-only dataset, aggregation, and trusted local checkpoint helpers."""
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from src.data.uav_data import select_samples, read_sample
from src.metrics.binary_segmentation import ConfusionMatrix
from src.training.segmentation_steps import masked_bce, masked_dice_bce

class CandidateDataset(Dataset):
    def __init__(self, manifest, data_dir, split, limit=None, crop=None):
        if split not in ('train_candidate','val_candidate'):
            raise ValueError('only train_candidate and val_candidate allowed')
        self.samples = sorted(select_samples(manifest, split, '2'), key=lambda s:s.filename)
        if limit is not None:
            if limit < 1: raise ValueError('limit must be positive')
            self.samples = self.samples[:limit]
        self.data_dir, self.crop = Path(data_dir), crop
    def __len__(self): return len(self.samples)
    def __getitem__(self, index):
        item = read_sample(self.samples[index], self.data_dir)
        x, y, v = item['image'], item['mask'], item['valid_mask']
        if self.crop:
            h,w = y.shape; n=self.crop
            if n%4 or n>min(h,w): raise ValueError('invalid crop')
            top,left=(h-n)//2,(w-n)//2
            x,y,v=x[:,top:top+n,left:left+n],y[top:top+n,left:left+n],v[top:top+n,left:left+n]
        return torch.from_numpy(x.copy()),torch.from_numpy(y.copy()).unsqueeze(0).float(),torch.from_numpy(v.copy()).unsqueeze(0)


def evaluate(model, batches, device, threshold=.5, loss_kind="bce", dice_smooth=1.0, bce_weight=1.0, dice_weight=1.0):
    """Pooled pixel metrics; restore prior model mode; no gradient or parameter update."""
    if not 0 <= threshold <= 1: raise ValueError('invalid threshold')
    was_training = model.training
    meter = ConfusionMatrix(); loss_sum=0.; valid_sum=0
    try:
        model.eval()
        with torch.no_grad():
            for x,y,v in batches:
                x,y,v=x.to(device),y.to(device),v.to(device)
                n=int(v.sum().item())
                if not n: raise ValueError('no valid pixels')
                z=model(x)
                if loss_kind == "bce": loss=masked_bce(z,y,v)
                elif loss_kind == "dice_bce": loss=masked_dice_bce(z,y,v,bce_weight,dice_weight,dice_smooth)
                else: raise ValueError("unknown loss_kind")
                if not torch.isfinite(loss): raise ValueError('nonfinite validation loss')
                loss_sum += float(loss)*n; valid_sum += n
                p=(torch.sigmoid(z)>=threshold).cpu().numpy(); t=y.cpu().numpy(); m=v.cpu().numpy()
                for a,b,c in zip(p,t,m): meter.update(a,b,c)
    finally:
        model.train(was_training)
    if not valid_sum: raise ValueError('empty validation')
    return {'loss':loss_sum/valid_sum, 'threshold':threshold, 'confusion':meter.result()}


def save_checkpoint(path, model, optimizer, epoch, config, best):
    state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'config':config,'best':best,
           'rng_python':random.getstate(),'rng_numpy':np.random.get_state(),'rng_torch':torch.get_rng_state(),
           'rng_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    torch.save(state,path)


def load_checkpoint(path, model, optimizer, device):
    """Only load trusted local checkpoints; pickle can execute code."""
    s=torch.load(path,map_location=device,weights_only=False)
    model.load_state_dict(s['model']); optimizer.load_state_dict(s['optimizer'])
    random.setstate(s['rng_python']); np.random.set_state(s['rng_numpy']); torch.set_rng_state(s['rng_torch'].cpu())
    if s['rng_cuda'] is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all([state.cpu() for state in s['rng_cuda']])
    return s
