"""Small two-level U-Net: RGB NCHW to one logit per pixel, no sigmoid."""
import torch
from torch import nn


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU())


class SmallUNet(nn.Module):
    """Height and width must be positive multiples of 4 (two pooling steps)."""
    def __init__(self, base_channels=8):
        super().__init__()
        if not isinstance(base_channels, int) or isinstance(base_channels, bool) or base_channels < 1:
            raise ValueError('base_channels must be a positive integer')
        n = base_channels
        self.enc1, self.enc2, self.bottleneck = block(3, n), block(n, 2*n), block(2*n, 4*n)
        self.pool = nn.MaxPool2d(2)
        self.up2, self.dec2 = nn.ConvTranspose2d(4*n, 2*n, 2, stride=2), block(4*n, 2*n)
        self.up1, self.dec1 = nn.ConvTranspose2d(2*n, n, 2, stride=2), block(2*n, n)
        self.head = nn.Conv2d(n, 1, 1)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 3 or x.shape[2] < 4 or x.shape[3] < 4 or x.shape[2] % 4 or x.shape[3] % 4:
            raise ValueError('input must be N x 3 x H x W with H,W >= 4 and multiples of 4')
        a = self.enc1(x); b = self.enc2(self.pool(a)); c = self.bottleneck(self.pool(b))
        return self.head(self.dec1(torch.cat((self.up1(self.dec2(torch.cat((self.up2(c), b), dim=1))), a), dim=1)))
