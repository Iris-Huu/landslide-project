"""Small CPU smoke test; default synthetic, explicit train candidates only for real mode."""
import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from src.data.uav_data import select_samples, read_sample, VALID_POLICY
from src.models.unet import SmallUNet
from src.training.segmentation_steps import masked_bce, train_step

MANIFEST = ROOT / 'data/interim/splits/moxitaidi_unified_spatial_manifest_v1.csv'
DATA = ROOT / 'data/raw/CAS/Moxitaidi_UAV_0.6m'


@dataclass(frozen=True)
class SmokeConfig:
    purpose: str = 'exploratory_smoke_test_only'
    formal_approval: str = 'unchanged; not granted by this smoke test'
    valid_policy: str = 'all_pixels_provisional'
    valid_limitation: str = 'valid region has not been verified'
    device: str = 'cpu'
    threads: int = 1
    seed: int = 17
    base_channels: int = 2
    crop_size: int = 128
    batch_size: int = 1
    learning_rate: float = 0.001
    max_updates: int = 2
    max_step_seconds: int = 60


def tensor_batch(sample, crop_size):
    image, mask, valid = sample['image'], sample['mask'], sample['valid_mask']
    if not isinstance(image, np.ndarray) or image.dtype != np.float32 or image.ndim != 3 or image.shape[0] != 3:
        raise ValueError('image must be float32 CHW RGB')
    if not isinstance(mask, np.ndarray) or mask.dtype != np.uint8 or mask.shape != image.shape[1:] or not np.isin(mask, (0, 1)).all():
        raise ValueError('mask must be uint8 HW, binary, same size as image')
    if not isinstance(valid, np.ndarray) or valid.dtype != np.bool_ or valid.shape != mask.shape:
        raise ValueError('valid_mask must be bool HW, same size as image')
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError('image values must be finite and in [0,1]')
    if crop_size < 4 or crop_size % 4 or any(x < crop_size for x in mask.shape):
        raise ValueError('crop size must be a multiple of 4 and fit image')
    top, left = (mask.shape[0] - crop_size)//2, (mask.shape[1] - crop_size)//2
    x = torch.from_numpy(image[:, top:top+crop_size, left:left+crop_size].copy()).unsqueeze(0).to('cpu')
    y = torch.from_numpy(mask[top:top+crop_size, left:left+crop_size].copy()).unsqueeze(0).unsqueeze(0).to(device='cpu', dtype=torch.float32)
    v = torch.from_numpy(valid[top:top+crop_size, left:left+crop_size].copy()).unsqueeze(0).unsqueeze(0).to('cpu')
    return x, y, v, {'top': top, 'left': left, 'height': crop_size, 'width': crop_size}


def run_update(model, optimizer, batch):
    x, y, v, crop = batch
    start = time.perf_counter()
    before_params = [p.detach().clone() for p in model.parameters()]
    with torch.no_grad():
        before = float(masked_bce(model(x), y, v))
    update_loss = train_step(model, optimizer, x, y, v)
    with torch.no_grad():
        logits = model(x)
        after = float(masked_bce(logits, y, v))
    if not np.isfinite([before, update_loss, after]).all() or not torch.isfinite(logits).all():
        raise ValueError('non-finite loss or logits')
    delta = max(float((p.detach() - old).abs().max()) for p, old in zip(model.parameters(), before_params))
    if not delta > 0:
        raise ValueError('no model parameter changed')
    return {'crop': crop, 'seconds': time.perf_counter()-start, 'loss_before': before,
            'update_loss': update_loss, 'loss_after': after, 'parameter_changed': True,
            'max_abs_parameter_change': delta, 'finite_logits_loss_gradients': True,
            'tensor_shapes': {'image': list(x.shape), 'target': list(y.shape), 'valid_mask': list(v.shape)},
            'tensor_dtypes': {'image': str(x.dtype), 'target': str(y.dtype), 'valid_mask': str(v.dtype)}}


def synthetic_gate():
    cfg = SmokeConfig(crop_size=16, max_updates=1)
    torch.manual_seed(cfg.seed)
    sample = {'image': np.full((3,16,16), .5, np.float32), 'mask': np.indices((16,16)).sum(axis=0).astype(np.uint8)%2,
              'valid_mask': np.ones((16,16), dtype=bool)}
    batch = tensor_batch(sample, cfg.crop_size)
    x,y,v,_ = batch
    assert x.shape == (1,3,16,16) and y.shape == v.shape == (1,1,16,16)
    assert x.dtype == y.dtype == torch.float32 and v.dtype == torch.bool
    try:
        masked_bce(torch.zeros_like(y), y, torch.zeros_like(v))
    except ValueError as exc:
        if 'no valid pixels' not in str(exc): raise
    else:
        raise AssertionError('all-invalid mask must fail')
    mask = torch.tensor([[[[True,False]]]])
    a = masked_bce(torch.tensor([[[[0.,0.]]]]), torch.zeros(1,1,1,2), mask)
    b = masked_bce(torch.tensor([[[[0.,100.]]]]), torch.zeros(1,1,1,2), mask)
    assert torch.equal(a,b), 'masked pixels affected loss'
    model = SmallUNet(cfg.base_channels).to('cpu')
    result = run_update(model, torch.optim.SGD(model.parameters(), lr=cfg.learning_rate), batch)
    log = {'mode': 'synthetic', 'config': asdict(cfg), 'step': result}
    for key in ('purpose','formal_approval','valid_policy','device','threads','seed','base_channels','crop_size','batch_size','learning_rate','max_updates'):
        assert key in log['config'], key
    assert log['step']['parameter_changed'] and log['step']['finite_logits_loss_gradients']
    return log


def git(*args):
    return subprocess.check_output(['git', '--no-optional-locks', *args], cwd=ROOT, text=True).strip()


def save(path, obj):
    # Dedicated new directory; keep a recoverable record after each stage.
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('synthetic','real-train'), default='synthetic')
    args = parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    cfg = SmokeConfig()
    assert cfg.valid_policy == VALID_POLICY
    run_dir = Path(tempfile.mkdtemp(prefix='uav_cpu_smoke_', dir=ROOT/'experiments'))
    out = run_dir/'run.json'
    code = [Path(__file__), ROOT/'src/data/uav_data.py', ROOT/'src/models/unet.py', ROOT/'src/training/segmentation_steps.py']
    record = {'started_utc': datetime.now(timezone.utc).isoformat(), 'mode': args.mode, 'config': asdict(cfg),
              'environment': {'python': sys.version, 'torch': torch.__version__, 'torch_cuda_build': torch.version.cuda,
                              'numpy': np.__version__, 'platform': platform.platform(), 'device': 'cpu', 'threads': torch.get_num_threads()},
              'git_head': git('rev-parse','HEAD'), 'git_status': git('status','--short','--untracked-files=all'),
              'code_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in code},
              'steps': [], 'status': 'started', 'checkpoint': None}
    save(out, record)
    try:
        record['synthetic_gate'] = synthetic_gate()
        record['status'] = 'synthetic_passed'; save(out, record)
        if args.mode == 'real-train':
            samples = sorted(select_samples(MANIFEST, 'train_candidate', component='2'), key=lambda s:s.filename)[:2]
            if len(samples) != 2: raise ValueError('expected at least two train candidates')
            record['selection_rule'] = 'first two train_candidate filenames in lexical order, UAV component 2'
            record['manifest_path'] = samples[0].manifest_path
            record['manifest_sha256'] = samples[0].manifest_sha256
            record['sample_names'] = [s.filename for s in samples]
            torch.manual_seed(cfg.seed)
            model = SmallUNet(cfg.base_channels).to('cpu')
            record['model_parameter_count'] = sum(p.numel() for p in model.parameters())
            optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate)
            save(out, record)
            for i, sample in enumerate(samples, 1):
                record['status'] = f'reading_and_updating_step_{i}'; save(out, record)
                start = time.perf_counter()
                data = read_sample(sample, DATA, valid_policy=cfg.valid_policy)
                batch = tensor_batch(data, cfg.crop_size)
                step = run_update(model, optimizer, batch)
                step.update({'step': i, 'filename': sample.filename, 'read_and_step_seconds': time.perf_counter()-start})
                record['steps'].append(step); save(out, record)
                if step['read_and_step_seconds'] > cfg.max_step_seconds:
                    raise TimeoutError('step exceeded 60 seconds; stopping further updates')
        record['status'] = 'completed'
    except Exception as exc:
        record['status'] = 'failed'; record['error'] = f'{type(exc).__name__}: {exc}'
        save(out, record); print(f'Failed; record: {out}', file=sys.stderr); raise
    finally:
        record['finished_utc'] = datetime.now(timezone.utc).isoformat(); save(out, record)
    print(json.dumps({'record': str(out), 'status': record['status'], 'steps': record['steps'], 'synthetic_gate': record['synthetic_gate']}, indent=2))

if __name__ == '__main__': main()
