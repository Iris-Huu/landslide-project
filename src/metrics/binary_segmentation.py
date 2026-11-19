"""Binary segmentation metrics using NumPy only.

Masks contain only 0/1 (or bool); optional valid_mask must be boolean.
Without valid_mask all pixels are counted, not necessarily verified valid.
Undefined ratios are None, serialized as JSON null (use allow_nan=False).
"""
from dataclasses import dataclass
import numpy as np


def _binary(value, name):
    arr = np.asarray(value)
    if arr.dtype.kind not in 'buif' or not np.all((arr == 0) | (arr == 1)):
        raise ValueError(f'{name} must contain only numeric or boolean 0 and 1')
    return arr.astype(bool, copy=False)


@dataclass
class ConfusionMatrix:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    samples: int = 0
    empty_samples: int = 0

    def update(self, prediction, target, valid_mask=None):
        pred = _binary(prediction, 'prediction')
        truth = _binary(target, 'target')
        if pred.shape != truth.shape:
            raise ValueError('prediction and target shapes must match')
        if valid_mask is None:
            valid = np.ones(pred.shape, dtype=bool)
        else:
            valid = np.asarray(valid_mask)
            if valid.dtype != np.bool_:
                raise ValueError('valid_mask must have boolean dtype')
            if valid.shape != pred.shape:
                raise ValueError('valid_mask shape must match prediction and target')
        self.tp += int(np.count_nonzero(valid & pred & truth))
        self.fp += int(np.count_nonzero(valid & pred & ~truth))
        self.fn += int(np.count_nonzero(valid & ~pred & truth))
        self.tn += int(np.count_nonzero(valid & ~pred & ~truth))
        self.samples += 1
        self.empty_samples += int(not valid.any())
        return self

    def result(self):
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        denominators = {'iou': tp + fp + fn, 'f1': 2 * tp + fp + fn,
                        'precision': tp + fp, 'recall': tp + fn}
        nums = {'iou': tp, 'f1': 2 * tp, 'precision': tp, 'recall': tp}
        return {'aggregation': 'pooled_pixels', 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                'valid_pixels': tp + fp + fn + tn, 'samples': self.samples,
                'empty_samples': self.empty_samples, 'denominators': denominators,
                'metrics': {k: nums[k] / d if d else None for k, d in denominators.items()}}
