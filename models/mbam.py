import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _choose_gn_groups(num_channels: int) -> int:
    for g in (8, 4, 2):
        if num_channels % g == 0:
            return g
    return 1


class SobelEdge(nn.Module):
    """Fixed Sobel operator for boundary/edge cues.

    Input:  (B, 1, H, W)
    Output: (B, 1, H, W) gradient magnitude
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

        gx = torch.tensor(
            [[-1.0, 0.0, 1.0],
             [-2.0, 0.0, 2.0],
             [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        gy = torch.tensor(
            [[-1.0, -2.0, -1.0],
             [0.0, 0.0, 0.0],
             [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        )
        weight = torch.stack([gx, gy], dim=0).unsqueeze(1)  # (2, 1, 3, 3)
        self.register_buffer("weight", weight, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grad = F.conv2d(x, self.weight.to(dtype=x.dtype), padding=1)
        gx, gy = grad[:, 0:1], grad[:, 1:2]
        mag = torch.sqrt(gx * gx + gy * gy + self.eps)
        return mag


class MBAM2D(nn.Module):
    """Multi-scale Boundary-aware Attention Module (MBAM).

    Two parallel branches:
    - Multi-scale fusion: 1x1, 3x3, dilated 3x3
    - Boundary-aware attention: feature->1ch, Sobel, conv, sigmoid

    Innovation (lightweight & stable): boundary-guided local scale weighting.
    Boundary map produces per-pixel soft weights over the 3 scales.

    Input/Output: (B, C, H, W)
    """

    def __init__(
        self,
        channels: int,
        branch_ratio: int = 4,
        dilation: int = 2,
        ca_ratio: int = 8,
        branch_mask: Sequence[int] = (1, 1, 1),
        use_boundary_branch: bool = True,
        use_scale_selection: bool = True,
        use_channel_attention: bool = True,
        use_edge_gate: bool = True,
        use_res_scale: bool = True,
    ):
        super().__init__()
        branch_channels = max(channels // branch_ratio, 8)

        if len(branch_mask) != 3:
            raise ValueError(f"branch_mask must have length 3, got {len(branch_mask)}")
        branch_mask = tuple(1 if int(v) else 0 for v in branch_mask)
        if sum(branch_mask) == 0:
            raise ValueError("branch_mask must enable at least one branch")

        self.branch_mask = branch_mask
        self.use_boundary_branch = bool(use_boundary_branch)
        self.use_scale_selection = bool(use_scale_selection) and self.use_boundary_branch
        self.use_channel_attention = bool(use_channel_attention)
        self.use_edge_gate = bool(use_edge_gate) and self.use_boundary_branch
        self.use_res_scale = bool(use_res_scale)

        def ms_block(kernel_size: int, dilation_: int = 1) -> nn.Sequential:
            padding = (kernel_size // 2) * dilation_
            return nn.Sequential(
                nn.Conv2d(channels, branch_channels, kernel_size, padding=padding, dilation=dilation_, bias=False),
                nn.GroupNorm(_choose_gn_groups(branch_channels), branch_channels),
                nn.SiLU(inplace=True),
            )

        self.ms_1x1 = ms_block(kernel_size=1)
        self.ms_3x3 = ms_block(kernel_size=3)
        self.ms_dilated_3x3 = ms_block(kernel_size=3, dilation_=dilation)

        self.boundary_reduce = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
        )
        self.sobel = SobelEdge()

        # Boundary attention: predict boundary from both feature cue and edge cue
        self.boundary_head = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_choose_gn_groups(8), 8),
            nn.SiLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # Boundary-guided scale weights (per-pixel softmax over 3 scales)
        self.scale_logits = nn.Conv2d(2, 3, kernel_size=1, bias=True)

        self.ms_fuse = nn.Sequential(
            nn.Conv2d(branch_channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(_choose_gn_groups(channels), channels),
            nn.SiLU(inplace=True),
        )

        # Lightweight channel attention on fused features
        ca_hidden = max(channels // ca_ratio, 8)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, ca_hidden, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(ca_hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # Learnable (zero-init) residual scaling for training stability
        self.res_scale = nn.Parameter(torch.zeros(1))
        self.edge_gate = nn.Parameter(torch.zeros(1))
        self.ca_gate = nn.Parameter(torch.zeros(1))

        self._eps = 1e-6

    def _merge_scales(self, f1: torch.Tensor, f2: torch.Tensor, f3: torch.Tensor, boundary_in: Optional[torch.Tensor]) -> torch.Tensor:
        feats = [f1, f2, f3]
        active_idx = [i for i, enabled in enumerate(self.branch_mask) if enabled]
        if len(active_idx) == 1:
            return feats[active_idx[0]]

        if self.use_scale_selection and boundary_in is not None:
            logits = self.scale_logits(boundary_in)
            sel = torch.cat([logits[:, i:i + 1] for i in active_idx], dim=1)
            w = F.softmax(sel, dim=1)
            ms = 0.0
            for j, i in enumerate(active_idx):
                ms = ms + w[:, j:j + 1] * feats[i]
            return ms

        ms = 0.0
        for i in active_idx:
            ms = ms + feats[i]
        return ms / float(len(active_idx))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        boundary_in = None
        edge_attn = None
        if self.use_boundary_branch:
            # boundary attention branch
            boundary_base = self.boundary_reduce(x)
            edge_mag = self.sobel(boundary_base)

            # normalize edge magnitude per-sample to stabilize logits across batches
            edge_mean = edge_mag.mean(dim=(2, 3), keepdim=True)
            edge_mag_n = edge_mag / (edge_mean + self._eps)
            boundary_in = torch.cat([boundary_base, edge_mag_n], dim=1)

            if self.use_edge_gate:
                edge_attn = self.boundary_head(boundary_in)  # (B,1,H,W)

        # multi-scale branch
        f1 = self.ms_1x1(x)
        f2 = self.ms_3x3(x)
        f3 = self.ms_dilated_3x3(x)

        ms = self._merge_scales(f1, f2, f3, boundary_in)

        ms = self.ms_fuse(ms)

        # channel attention (gated, zero-init for stability)
        if self.use_channel_attention:
            ms = ms * (1.0 + self.ca_gate * self.ca(ms))

        # edge-gated residual enhancement
        if self.use_edge_gate and edge_attn is not None:
            ms = ms * (1.0 + self.edge_gate * edge_attn)

        if self.use_res_scale:
            out = x + self.res_scale * ms
        else:
            out = x + ms
        return out


class MBAMToken(nn.Module):
    """MBAM wrapper for VM-UNet tensors in BHWC format."""

    def __init__(
        self,
        channels: int,
        branch_ratio: int = 4,
        dilation: int = 2,
        ca_ratio: int = 8,
        branch_mask: Sequence[int] = (1, 1, 1),
        use_boundary_branch: bool = True,
        use_scale_selection: bool = True,
        use_channel_attention: bool = True,
        use_edge_gate: bool = True,
        use_res_scale: bool = True,
    ):
        super().__init__()
        self.mbam = MBAM2D(
            channels=channels,
            branch_ratio=branch_ratio,
            dilation=dilation,
            ca_ratio=ca_ratio,
            branch_mask=branch_mask,
            use_boundary_branch=use_boundary_branch,
            use_scale_selection=use_scale_selection,
            use_channel_attention=use_channel_attention,
            use_edge_gate=use_edge_gate,
            use_res_scale=use_res_scale,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, C)
        x_nchw = x.permute(0, 3, 1, 2).contiguous()
        y = self.mbam(x_nchw)
        y = y.permute(0, 2, 3, 1).contiguous()
        return y
