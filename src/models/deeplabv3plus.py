"""DeepLabV3+ via segmentation-models-pytorch, with no implicit weight downloads."""
import hashlib
from pathlib import Path
import torch
import segmentation_models_pytorch as smp


def build_deeplabv3plus(encoder='resnet18', encoder_weights_path=None):
    """Return model and weight provenance. Optional path is a trusted encoder-only state dict.

    The caller is responsible for licensing and provenance of any local weights.
    No built-in pretrained weights are requested or downloaded.
    """
    if not isinstance(encoder, str) or not encoder.strip():
        raise ValueError('encoder must be a nonempty SMP encoder name')
    model=smp.DeepLabV3Plus(encoder_name=encoder, encoder_weights=None,
                            in_channels=3, classes=1, activation=None,
                            encoder_output_stride=16, decoder_atrous_rates=(12,24,36))
    provenance={'status':'not_used_random_initialization','source':None,'sha256':None}
    if encoder_weights_path is not None:
        path=Path(encoder_weights_path)
        if not path.is_file(): raise FileNotFoundError(path)
        state=torch.load(path,map_location='cpu',weights_only=True)
        model.encoder.load_state_dict(state,strict=True)
        provenance={'status':'local_encoder_state_dict','source':str(path.resolve()),
                    'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    return model, provenance
