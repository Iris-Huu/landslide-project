"""Conservative UAV valid-pixel mask, version 1."""

import numpy as np
from scipy import ndimage


def valid_mask_v1(image, label):
    """
    image: RGB array with shape (3, H, W), uint8.
    label: binary array with shape (H, W).

    Returns a boolean array where True means the pixel is retained.
    Black regions are connected using four-neighbour connectivity.
    """
    image = np.asarray(image)
    label = np.asarray(label)

    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("image must have shape (3, H, W)")
    if image.dtype != np.uint8:
        raise ValueError("image must have dtype uint8")
    if label.shape != image.shape[1:]:
        raise ValueError("label shape must match image height and width")
    if label.size == 0 or not np.isin(label, (0, 1)).all():
        raise ValueError("label must be non-empty and contain only 0 and 1")

    black = np.all(image == 0, axis=0)

    edge_seed = np.zeros(black.shape, dtype=bool)
    edge_seed[0, :] = black[0, :]
    edge_seed[-1, :] = black[-1, :]
    edge_seed[:, 0] = black[:, 0]
    edge_seed[:, -1] = black[:, -1]

    structure = ndimage.generate_binary_structure(2, 1)
    edge_black = ndimage.binary_propagation(
        edge_seed,
        structure=structure,
        mask=black,
    )

    ignore = edge_black & (label == 0)
    return ~ignore
