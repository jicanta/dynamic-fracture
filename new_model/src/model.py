# src/model.py
"""FractureTAU: SimVP-style spatiotemporal model with a TAU translator.

Layout
  spatial encoder  (shared across frames, 2x downsampling twice)
  temporal translator (stack of Temporal Attention Unit blocks operating on
                       all T latent frames merged into channels)
  spatial decoder  (upsamples back, skip connection from the encoder)

References: SimVP (CVPR 2022), TAU "Temporal Attention Unit" (CVPR 2023).
Input  : (B, T, C, H, W) float32
Output : (B, T, 1, H, W) logits (next-frame fracture mask for each step,
         i.e. output t corresponds to input frame t shifted by +1)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(channels: int) -> nn.GroupNorm:
    groups = 8
    while channels % groups != 0:
        groups //= 2
    return nn.GroupNorm(max(groups, 1), channels)


class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask = torch.rand(x.shape[0], 1, 1, 1, device=x.device, dtype=x.dtype) < keep
        return x * mask / keep


class ConvSC(nn.Module):
    """conv3x3 + GroupNorm + SiLU; optional 2x down (stride) or 2x up (bilinear)."""

    def __init__(self, c_in: int, c_out: int, *, down: bool = False, up: bool = False):
        super().__init__()
        self.up = up
        stride = 2 if down else 1
        self.conv = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1)
        self.norm = _gn(c_out)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.up:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.act(self.norm(self.conv(x)))


class Encoder(nn.Module):
    def __init__(self, c_in: int, hid: int):
        super().__init__()
        self.b1 = ConvSC(c_in, hid)            # full res (skip source)
        self.b2 = ConvSC(hid, hid, down=True)  # /2
        self.b3 = ConvSC(hid, hid)
        self.b4 = ConvSC(hid, hid, down=True)  # /4

    def forward(self, x: torch.Tensor):
        s = self.b1(x)
        z = self.b4(self.b3(self.b2(s)))
        return z, s


class Decoder(nn.Module):
    def __init__(self, hid: int):
        super().__init__()
        self.b1 = ConvSC(hid, hid)
        self.b2 = ConvSC(hid, hid, up=True)    # /2
        self.b3 = ConvSC(hid, hid)
        self.b4 = ConvSC(hid, hid, up=True)    # full res
        self.read = nn.Sequential(
            ConvSC(2 * hid, hid),
            nn.Conv2d(hid, 1, 1),
        )

    def forward(self, z: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.b4(self.b3(self.b2(self.b1(z))))
        return self.read(torch.cat([x, skip], dim=1))


class TAUBlock(nn.Module):
    """Temporal Attention Unit: large-kernel static attention (depthwise 5x5 +
    dilated depthwise 7x7) modulated by a dynamic channel gate, plus a 1x1 MLP."""

    def __init__(self, dim: int, drop_path: float = 0.0, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = _gn(dim)
        self.dw = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.dw_d = nn.Conv2d(dim, dim, 7, padding=9, dilation=3, groups=dim)
        self.pw = nn.Conv2d(dim, dim, 1)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, max(dim // 4, 8), 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(dim // 4, 8), dim, 1),
            nn.Sigmoid(),
        )
        self.proj = nn.Conv2d(dim, dim, 1)

        hidden = int(dim * mlp_ratio)
        self.norm2 = _gn(dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, dim, 1),
        )
        self.dp1 = DropPath(drop_path)
        self.dp2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.norm1(x)
        attn = self.pw(self.dw_d(self.dw(v)))      # static large-kernel attention
        attn = attn * self.gate(v)                 # dynamic channel attention
        x = x + self.dp1(self.proj(attn * v))
        x = x + self.dp2(self.mlp(self.norm2(x)))
        return x


class Translator(nn.Module):
    def __init__(self, t_steps: int, hid_s: int, hid_t: int, n_blocks: int,
                 drop_path: float = 0.0):
        super().__init__()
        c = t_steps * hid_s
        self.inp = nn.Conv2d(c, hid_t, 1)
        dprs = torch.linspace(0, drop_path, n_blocks).tolist()
        self.blocks = nn.Sequential(*[TAUBlock(hid_t, dp) for dp in dprs])
        self.out = nn.Conv2d(hid_t, c, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.out(self.blocks(self.inp(z)))


class PlainBlock(nn.Module):
    """Plain residual conv block (no temporal attention): conv3x3 + GroupNorm +
    GELU, residual add. The ablation-row-5 (D-07) counterpart to TAUBlock —
    matches its dim-preserving (dim -> dim) contract so it is a drop-in inside the
    same in/out 1x1 convs of the translator bottleneck."""

    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()
        self.norm = _gn(dim)
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)
        self.act = nn.GELU()
        self.dp = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dp(self.conv(self.act(self.norm(x))))


class PlainTranslator(nn.Module):
    """Plain-conv translator for ablation row 5 (D-07): IDENTICAL in/out 1x1 conv
    shapes to ``Translator`` (so the encoder/decoder around it are unchanged), but
    stacks ``PlainBlock``s instead of ``TAUBlock``s — i.e. NO temporal attention.
    Because its block parameters differ from the TAU translator, a run with
    translator_type='plain' CANNOT warm-start from a TAU frozen Stage-1
    (``--init-from`` would shape-mismatch); it needs its own full train (plan 07)."""

    def __init__(self, t_steps: int, hid_s: int, hid_t: int, n_blocks: int,
                 drop_path: float = 0.0):
        super().__init__()
        c = t_steps * hid_s
        self.inp = nn.Conv2d(c, hid_t, 1)
        dprs = torch.linspace(0, drop_path, n_blocks).tolist()
        self.blocks = nn.Sequential(*[PlainBlock(hid_t, dp) for dp in dprs])
        self.out = nn.Conv2d(hid_t, c, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.out(self.blocks(self.inp(z)))


class FractureTAU(nn.Module):
    def __init__(self, *, c_in: int, t_steps: int, hid_s: int = 64,
                 hid_t: int = 384, n_temporal: int = 6, drop_path: float = 0.05,
                 head_type: str = "sigmoid", translator_type: str = "tau"):
        super().__init__()
        self.t_steps = t_steps
        self.hid_s = hid_s
        self.head_type = head_type           # 'sigmoid' (logits) | 'monotone_delta' (prob)
        self.translator_type = translator_type  # 'tau' (TAU blocks) | 'plain' (plain conv) — D-07
        self.encoder = Encoder(c_in, hid_s)
        # CRITICAL (D-07 / RESEARCH Pitfall 5): the plain translator has DIFFERENT
        # parameters than the TAU translator, so a --init-from <tau_stage1> warm-start
        # would shape-mismatch. Ablation row 5 needs its OWN full Stage-1+Stage-2 train
        # (plan 07), NOT the shared-frozen-Stage-1 budget trick used for TAU sweeps.
        if translator_type == "tau":
            self.translator = Translator(t_steps, hid_s, hid_t, n_temporal, drop_path)
        else:
            self.translator = PlainTranslator(t_steps, hid_s, hid_t, n_temporal, drop_path)
        self.decoder = Decoder(hid_s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        assert T == self.t_steps, f"expected T={self.t_steps}, got {T}"

        # Capture the no-healing input mask (channel 0) BEFORE the reshape below
        # so the monotone-Δ head can recover prev = x_orig[:, :, 0:1].
        x_orig = x

        # pad H, W to multiples of 4 (two 2x downsamples)
        ph = (-H) % 4
        pw = (-W) % 4
        x = x.reshape(B * T, C, H, W)
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        Hp, Wp = H + ph, W + pw

        z, skip = self.encoder(x)
        _, _, h, w = z.shape
        z = z.reshape(B, T * self.hid_s, h, w)
        z = self.translator(z)
        z = z.reshape(B * T, self.hid_s, h, w)

        out = self.decoder(z, skip)                # (B*T, 1, Hp, Wp)
        out = out[..., :H, :W]
        out = out.reshape(B, T, 1, H, W)           # raw logits
        if self.head_type == "monotone_delta":
            prev = x_orig[:, :, 0:1]               # no-healing state, input channel 0
            q = torch.sigmoid(out)                 # non-negative increment in [0, 1]
            return prev + (1.0 - prev) * q         # probabilistic-OR: >= prev, in [0,1] — PROB
        return out                                  # sigmoid head: logits (unchanged)


def build_model(cfg, c_in: int) -> FractureTAU:
    return FractureTAU(
        c_in=c_in,
        t_steps=cfg.sequence_length,
        hid_s=cfg.hid_s,
        hid_t=cfg.hid_t,
        n_temporal=cfg.n_temporal,
        drop_path=cfg.drop_path,
        head_type=cfg.head_type,
        translator_type=cfg.translator_type,
    )
