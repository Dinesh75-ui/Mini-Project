import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building Blocks
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) × 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscale: MaxPool → DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class AttentionGate(nn.Module):
    """
    Additive attention gate (Oktay et al., 2018).

    Focuses the decoder on spatially relevant regions in the skip connection,
    suppressing irrelevant background activations. This is the single most
    impactful quality improvement for colorization without changing resolution.

    g  : gating signal from the decoder (coarser, more semantic)
    x  : skip connection from the encoder (finer spatial detail)
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # Upsample g1 to match x1 spatial size if needed
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi   # attended skip connection


class Up(nn.Module):
    """Upsample → Attention Gate on skip → DoubleConv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up   = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

        # Attention gate: g = upsampled decoder, x = skip
        F_g   = in_channels // 2 if bilinear else in_channels // 2
        F_l   = out_channels
        F_int = out_channels // 2
        self.attn = AttentionGate(F_g=F_g, F_l=F_l, F_int=max(F_int, 1))

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Pad x1 to match x2 size
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                         diffY // 2, diffY - diffY // 2])

        # Apply attention gate on skip connection x2
        x2 = self.attn(g=x1, x=x2)

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# ---------------------------------------------------------------------------
# UNet with Attention Gates
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=2, bilinear=False):
        """
        n_channels : input channels  (1 for L channel in LAB color space)
        n_classes  : output channels (2 for a, b channels in LAB color space)
        bilinear   : use bilinear upsampling instead of transposed conv
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes  = n_classes
        self.bilinear   = bilinear

        # Encoder
        self.inc   = DoubleConv(n_channels, 64)
        self.down1 = Down(64,  128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor     = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # Decoder (each Up block includes an AttentionGate on the skip)
        self.up1 = Up(1024, 512  // factor, bilinear)
        self.up2 = Up(512,  256  // factor, bilinear)
        self.up3 = Up(256,  128  // factor, bilinear)
        self.up4 = Up(128,  64,             bilinear)

        # Output: Tanh scales to [-1, 1] matching the ab normalisation
        self.outc = nn.Sequential(
            OutConv(64, n_classes),
            nn.Tanh()
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)
        return self.outc(x)


if __name__ == '__main__':
    model = UNet(n_channels=1, n_classes=2)
    x = torch.randn(1, 1, 64, 64)
    y = model(x)
    print(f"Input  shape: {x.shape}")
    print(f"Output shape: {y.shape}")   # expected: (1, 2, 64, 64)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")
