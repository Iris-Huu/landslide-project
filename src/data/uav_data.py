"""UAV candidate TIFF reader, independent of training frameworks.

Output: RGB float32 CHW in [0,1], mask uint8 HW in {0,1}, valid bool HW.
All-pixels-provisional is an assumption, not a verified valid-data mask.
"""
import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
import warnings

import numpy as np
import rasterio
from rasterio.enums import ColorInterp
from rasterio.errors import NotGeoreferencedWarning

ALLOWED_SPLITS = ('train_candidate', 'val_candidate')
VALID_POLICY = 'all_pixels_provisional'
VALID_LIMITATION = 'All pixels included provisionally; valid area has not been verified.'


@dataclass(frozen=True)
class Sample:
    filename: str
    split: str
    component: str
    manifest_path: str
    manifest_sha256: str


def select_samples(manifest_path, split, component='2'):
    """Select only one allowed UAV candidate split; reject duplicates and unsafe names."""
    if split not in ALLOWED_SPLITS:
        raise ValueError(f'Unsupported split {split!r}; only {ALLOWED_SPLITS} allowed')
    component = str(component)
    if not component or not component.isdecimal() or int(component) < 1:
        raise ValueError(f'Invalid component {component!r}')
    path = Path(manifest_path).resolve()
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    # Decode once so the selected records and hash always describe the same bytes.
    import io
    reader = csv.DictReader(io.StringIO(payload.decode('utf-8-sig'), newline=''))
    fields = reader.fieldnames or []
    required = {'platform', 'component', 'split', 'filename'}
    if len(fields) != len(set(fields)) or not required.issubset(fields):
        raise ValueError(f'Manifest missing required or contains duplicate fields: {fields}')
    rows = [r for r in reader if r['platform'] == 'UAV' and r['component'] == component and r['split'] in ALLOWED_SPLITS]
    names = [r['filename'] for r in rows]
    if any(not n or Path(n).name != n or n in ('.', '..') for n in names):
        raise ValueError('Manifest has unsafe or empty filename')
    if len(names) != len(set(names)):
        raise ValueError('Manifest has duplicate UAV candidate filename within component')
    result = [Sample(r['filename'], r['split'], component, str(path), digest) for r in rows if r['split'] == split]
    if not result:
        raise ValueError(f'No UAV {split} samples in component {component}')
    return result


def read_sample(sample, data_dir, valid_policy=VALID_POLICY):
    """Read a selected sample; paths are <data_dir>/{img,mask}/<filename>."""
    if not isinstance(sample, Sample) or sample.split not in ALLOWED_SPLITS:
        raise ValueError('sample must be an allowed selected Sample')
    name = sample.filename
    if not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError(f'{name!r}: invalid filename')
    if valid_policy != VALID_POLICY:
        raise ValueError(f'{name}: unsupported valid-pixel policy {valid_policy!r}')
    arrays = {}
    for kind, count in (('img', 3), ('mask', 1)):
        path = Path(data_dir) / kind / name
        if not path.is_file():
            raise ValueError(f'{name}: missing {kind} file: {path}')
        try:
            with warnings.catch_warnings(), rasterio.Env(GDAL_PAM_ENABLED='NO'):
                warnings.simplefilter('ignore', NotGeoreferencedWarning)
                with rasterio.open(path) as ds:
                    if ds.count != count:
                        raise ValueError(f'{kind} expected {count} channels, got {ds.count}')
                    if any(x != 'uint8' for x in ds.dtypes):
                        raise ValueError(f'{kind} expected uint8 dtype, got {ds.dtypes}')
                    if kind == 'img' and ds.colorinterp != (ColorInterp.red, ColorInterp.green, ColorInterp.blue):
                        raise ValueError(f'img RGB color interpretation required, got {ds.colorinterp}')
                    arrays[kind] = ds.read()
        except Exception as exc:
            raise ValueError(f'{name}: {kind} read/validation failed: {exc}') from exc
    image, mask = arrays['img'], arrays['mask'][0]
    if image.shape[1:] != mask.shape:
        raise ValueError(f'{name}: image and mask dimensions differ: {image.shape[1:]} vs {mask.shape}')
    if not np.isin(mask, (0, 1)).all():
        raise ValueError(f'{name}: mask labels must be 0/1, got {np.unique(mask).tolist()}')
    return {'image': image.astype(np.float32) / np.float32(255), 'mask': mask,
            'valid_mask': np.ones(mask.shape, dtype=bool),
            'metadata': {'filename': name, 'split': sample.split, 'platform': 'UAV', 'component': sample.component,
                         'manifest_path': sample.manifest_path, 'manifest_sha256': sample.manifest_sha256,
                         'normalization': 'uint8_divide_255_float32', 'image_layout': 'CHW RGB',
                         'mask_layout': 'HW uint8 0/1', 'valid_policy': VALID_POLICY,
                         'valid_limitation': VALID_LIMITATION}}
