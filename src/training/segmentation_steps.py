"""Reusable tensor steps; caller decides data source and batching."""
from dataclasses import dataclass
import torch
from torch.nn import functional as F
from src.metrics.binary_segmentation import ConfusionMatrix


@dataclass(frozen=True)
class ExperimentConfig:
    base_channels: int = 8
    learning_rate: float = 0.001
    seed: int = 17
    threshold: float = 0.5
    image_height: int = 32
    image_width: int = 32
    valid_policy: str = 'all_pixels_provisional'


def masked_bce(logits, target, valid):
    if logits.shape != target.shape or logits.shape != valid.shape or logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError('logits, target and valid must have matching N x 1 x H x W shapes')
    if valid.dtype != torch.bool:
        raise ValueError('valid mask must be boolean')
    if not torch.all(torch.isfinite(logits)):
        raise ValueError('logits must be finite')
    if not torch.all((target == 0) | (target == 1)):
        raise ValueError('target must contain only 0/1')
    if not torch.any(valid):
        raise ValueError('no valid pixels in batch')
    return F.binary_cross_entropy_with_logits(logits[valid], target[valid].to(logits.dtype))


def train_step(model, optimizer, images, target, valid):
    model.train(); optimizer.zero_grad(set_to_none=True)
    logits = model(images)
    loss = masked_bce(logits, target, valid)
    loss.backward()
    for parameter in model.parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise ValueError("non-finite gradient; optimizer update aborted")
    optimizer.step()
    return float(loss.detach().cpu())


def metrics_from_logits(logits, target, valid, threshold=0.5):
    if not 0 <= threshold <= 1:
        raise ValueError('threshold must be between 0 and 1')
    loss = masked_bce(logits, target, valid)
    pred = (torch.sigmoid(logits) >= threshold).detach().cpu().numpy()
    truth = target.detach().cpu().numpy()
    mask = valid.detach().cpu().numpy()
    meter = ConfusionMatrix()
    for p, t, m in zip(pred, truth, mask): meter.update(p, t, m)
    return {'loss': float(loss.detach().cpu()), 'threshold': threshold, 'confusion': meter.result()}


def validation_step(model, images, target, valid, threshold=0.5):
    model.eval()
    with torch.no_grad():
        return metrics_from_logits(model(images), target, valid, threshold)
