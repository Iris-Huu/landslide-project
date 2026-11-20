"""SMP SegFormer, random by default; never implicitly download weights."""
import hashlib
from pathlib import Path
import segmentation_models_pytorch as smp
import torch


def build_segformer(encoder='mit_b2',encoder_weights_path=None):
    if not isinstance(encoder,str) or not encoder.strip(): raise ValueError('encoder must be nonempty')
    model=smp.Segformer(encoder_name=encoder,encoder_weights=None,in_channels=3,classes=1,activation=None,upsampling=4)
    info={'status':'not_used_random_initialization','source':None,'sha256':None}
    if encoder_weights_path is not None:
        path=Path(encoder_weights_path)
        if not path.is_file(): raise FileNotFoundError(path)
        model.encoder.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
        info={'status':'local_encoder_state_dict','source':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    return model,info
