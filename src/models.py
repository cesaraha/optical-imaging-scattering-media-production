# src/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── building blocks ───────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv → InstanceNorm → SiLU → Dropout2d"""
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.SiLU(),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    """Two ConvBlocks with a residual connection."""
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.conv1    = ConvBlock(in_ch,  out_ch, dropout)
        self.conv2    = ConvBlock(out_ch, out_ch, dropout)
        self.skip     = nn.Conv2d(in_ch, out_ch, 1, bias=False) \
                        if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.conv2(self.conv1(x)) + self.skip(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.SiLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.se(x).view(x.shape[0], x.shape[1], 1, 1)
        return x * w


class AttentionGate(nn.Module):
    """
    Spatial attention gate from Oktay et al. 2018.
    g = gating signal (from decoder)
    x = skip connection (from encoder)
    """
    def __init__(self, f_g: int, f_x: int, f_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, 1, bias=False),
            nn.InstanceNorm2d(f_int, affine=True),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(f_x, f_int, 1, bias=False),
            nn.InstanceNorm2d(f_int, affine=True),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, 1, bias=False),
            nn.InstanceNorm2d(1, affine=True),
            nn.Sigmoid(),
        )

    def forward(self, g, x):
        # g may be smaller than x if spatial dims differ — upsample to match
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode="bilinear",
                              align_corners=False)
        return x * self.psi(F.silu(self.W_g(g) + self.W_x(x)))


# ── encoder / decoder helpers ─────────────────────────────────────────────────

class Encoder(nn.Module):
    """Shared encoder for all U-Net variants."""
    def __init__(self, channels: list, block, dropout: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.pools  = nn.ModuleList()
        in_ch = 1
        for out_ch in channels:
            self.blocks.append(block(in_ch, out_ch, dropout))
            self.pools.append(nn.MaxPool2d(2))
            in_ch = out_ch

    def forward(self, x):
        skips = []
        for blk, pool in zip(self.blocks, self.pools):
            x = blk(x)
            skips.append(x)
            x = pool(x)
        return x, skips


# ── U-Net v1: baseline ────────────────────────────────────────────────────────

class UNetV1(nn.Module):
    """
    Baseline U-Net — faithful to thesis v5.
    ConvBlock + InstanceNorm + SiLU + Concatenate skips.
    channels default: [32, 64, 128, 256]
    """
    def __init__(self, channels: list = [32, 64, 128, 256],
                 dropout: float = 0.0):
        super().__init__()
        self.encoder = Encoder(channels, ConvBlock, dropout)

        # bottleneck
        self.bottleneck = ConvBlock(channels[-1], channels[-1] * 2, dropout)

        # decoder
        self.ups    = nn.ModuleList()
        self.dec    = nn.ModuleList()
        dec_chs     = [channels[-1] * 2] + list(reversed(channels))
        for i in range(len(channels)):
            self.ups.append(nn.ConvTranspose2d(dec_chs[i], dec_chs[i+1], 2, stride=2))
            self.dec.append(ConvBlock(dec_chs[i+1] * 2, dec_chs[i+1], dropout))

        self.head = nn.Conv2d(channels[0], 1, 1)

    def forward(self, x):
        x, skips = self.encoder(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.ups, self.dec, reversed(skips)):
            x = up(x)
            x = dec(torch.cat([x, skip], dim=1))
        return self.head(x)


# ── U-Net v2: residual ────────────────────────────────────────────────────────

class UNetV2(nn.Module):
    """
    Residual U-Net — ResidualBlock + Concatenate skips.
    """
    def __init__(self, channels: list = [32, 64, 128, 256],
                 dropout: float = 0.0):
        super().__init__()
        self.encoder    = Encoder(channels, ResidualBlock, dropout)
        self.bottleneck = ResidualBlock(channels[-1], channels[-1] * 2, dropout)

        self.ups = nn.ModuleList()
        self.dec = nn.ModuleList()
        dec_chs  = [channels[-1] * 2] + list(reversed(channels))
        for i in range(len(channels)):
            self.ups.append(nn.ConvTranspose2d(dec_chs[i], dec_chs[i+1], 2, stride=2))
            self.dec.append(ResidualBlock(dec_chs[i+1] * 2, dec_chs[i+1], dropout))

        self.head = nn.Conv2d(channels[0], 1, 1)

    def forward(self, x):
        x, skips = self.encoder(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.ups, self.dec, reversed(skips)):
            x = up(x)
            x = dec(torch.cat([x, skip], dim=1))
        return self.head(x)


# ── U-Net v3: attention gates ─────────────────────────────────────────────────

class UNetV3(nn.Module):
    """
    Attention U-Net — spatial attention gates at skip connections (Oktay 2018).
    """
    def __init__(self, channels: list = [32, 64, 128, 256],
                 dropout: float = 0.0):
        super().__init__()
        self.encoder    = Encoder(channels, ConvBlock, dropout)
        self.bottleneck = ConvBlock(channels[-1], channels[-1] * 2, dropout)

        self.ups      = nn.ModuleList()
        self.dec      = nn.ModuleList()
        self.att      = nn.ModuleList()
        dec_chs       = [channels[-1] * 2] + list(reversed(channels))
        for i in range(len(channels)):
            self.ups.append(nn.ConvTranspose2d(dec_chs[i], dec_chs[i+1], 2, stride=2))
            self.att.append(AttentionGate(
                f_g   = dec_chs[i+1],
                f_x   = dec_chs[i+1],
                f_int = dec_chs[i+1] // 2,
            ))
            self.dec.append(ConvBlock(dec_chs[i+1] * 2, dec_chs[i+1], dropout))

        self.head = nn.Conv2d(channels[0], 1, 1)

    def forward(self, x):
        x, skips = self.encoder(x)
        x = self.bottleneck(x)
        for up, att, dec, skip in zip(self.ups, self.att, self.dec,
                                       reversed(skips)):
            g = up(x)
            x = dec(torch.cat([g, att(g, skip)], dim=1))
        return self.head(x)


# ── U-Net v4: SE blocks ───────────────────────────────────────────────────────

class SEConvBlock(nn.Module):
    """ConvBlock followed by SE channel attention."""
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, dropout)
        self.se   = SEBlock(out_ch)

    def forward(self, x):
        return self.se(self.conv(x))


class UNetV4(nn.Module):
    """
    SE U-Net — channel attention via Squeeze-and-Excitation blocks.
    """
    def __init__(self, channels: list = [32, 64, 128, 256],
                 dropout: float = 0.0):
        super().__init__()
        self.encoder    = Encoder(channels, SEConvBlock, dropout)
        self.bottleneck = SEConvBlock(channels[-1], channels[-1] * 2, dropout)

        self.ups = nn.ModuleList()
        self.dec = nn.ModuleList()
        dec_chs  = [channels[-1] * 2] + list(reversed(channels))
        for i in range(len(channels)):
            self.ups.append(nn.ConvTranspose2d(dec_chs[i], dec_chs[i+1], 2, stride=2))
            self.dec.append(SEConvBlock(dec_chs[i+1] * 2, dec_chs[i+1], dropout))

        self.head = nn.Conv2d(channels[0], 1, 1)

    def forward(self, x):
        x, skips = self.encoder(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.ups, self.dec, reversed(skips)):
            x = up(x)
            x = dec(torch.cat([x, skip], dim=1))
        return self.head(x)


# ── Dense v1 ──────────────────────────────────────────────────────────────────

class DenseV1(nn.Module):
    """
    CNN encoder → Flatten → Dense output.
    Encoder: 32→64→128, two MaxPool steps (128→64→32).
    Dense: 128*32*32 → 128*128 pixels.
    """
    def __init__(self, channels: list = [32, 64, 128],
                 image_size: int = 128,
                 dropout: float = 0.0):
        super().__init__()
        self.image_size = image_size
        n_pools         = 2
        feat_size       = image_size // (2 ** n_pools)   # 32

        self.encoder = nn.Sequential(
            ConvBlock(1,           channels[0], dropout),
            nn.MaxPool2d(2),
            ConvBlock(channels[0], channels[1], dropout),
            nn.MaxPool2d(2),
            ConvBlock(channels[1], channels[2], dropout),
        )

        flat_dim = channels[2] * feat_size * feat_size   # 128*32*32 = 131072

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, image_size * image_size),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x.view(x.shape[0], 1, self.image_size, self.image_size)


# ── model factory ─────────────────────────────────────────────────────────────

MODELS = {
    "unet_v1": UNetV1,
    "unet_v2": UNetV2,
    "unet_v3": UNetV3,
    "unet_v4": UNetV4,
    "dense_v1": DenseV1,
}


def build_model(version: str, channels: list, dropout: float,
                image_size: int = 128) -> nn.Module:
    assert version in MODELS, \
        f"Unknown model version: {version}. Choose from {list(MODELS.keys())}"
    if version == "dense_v1":
        return MODELS[version](
            channels   = channels[:3],
            image_size = image_size,
            dropout    = dropout,
        )
    return MODELS[version](channels=channels, dropout=dropout)