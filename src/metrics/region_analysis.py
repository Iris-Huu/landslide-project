"""Reusable spatial region and calibration accounting for binary masks."""
import numpy as np
from src.metrics.binary_segmentation import ConfusionMatrix


def center_mask(shape, size=256):
    if len(shape)!=2 or size<=0 or size>min(shape): raise ValueError('invalid center crop')
    out=np.zeros(shape,dtype=bool); top=(shape[0]-size)//2; left=(shape[1]-size)//2
    out[top:top+size,left:left+size]=True
    return out


def regional_counts(pred, target, center, valid):
    if center.dtype!=np.bool_ or valid.dtype!=np.bool_ or pred.shape!=target.shape or center.shape!=pred.shape or valid.shape!=pred.shape:
        raise ValueError('matching shapes and boolean masks required')
    output={k:ConfusionMatrix().update(pred,target,mask).result() for k,mask in [('full',valid),('center',valid & center),('outside',valid & ~center)]}
    for k in ('tp','fp','fn','tn','valid_pixels'):
        if output['full'][k] != output['center'][k]+output['outside'][k]: raise AssertionError('region count mismatch')
    return output


def calibration_bins(prob, target, valid, bins=10):
    if prob.shape!=target.shape or prob.shape!=valid.shape or valid.dtype!=np.bool_: raise ValueError('invalid shapes or mask')
    if not np.all(np.isfinite(prob)) or np.any((prob<0)|(prob>1)) or not np.all((target==0)|(target==1)): raise ValueError('invalid probability or label')
    p=prob[valid]; t=target[valid]; idx=np.minimum((p*bins).astype(int),bins-1)
    result=[]
    for i in range(bins):
        selected=idx==i; count=int(selected.sum())
        result.append({'bin':i,'lower':i/bins,'upper':(i+1)/bins,'count':count,
                       'probability_sum':float(p[selected].sum(dtype=np.float64)), 'positive_pixels':int(t[selected].sum())})
    return result
