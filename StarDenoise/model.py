"""PyTorch reproduction of the SetiAstro Cosmic Clarity AI 3.6 denoiser.

The architecture follows SetiAstroCosmicClarity_denoise.py from the upstream
MIT-licensed project.  It is intentionally kept independent of the original
GUI so it can be embedded in a larger image-processing pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv2(self.relu(self.conv1(x))) + x)


class CosmicClarityDenoiseCNN(nn.Module):
    """Full-resolution residual encoder/decoder used by AI 3.6.

    Despite the encoder/decoder names, the network does not spatially
    downsample. Dilated convolutions and skip concatenations preserve faint
    stars and small-scale structure while increasing the receptive field.
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder1 = self._encoder(3, 16)
        self.encoder2 = self._encoder(16, 32)
        self.encoder3 = self._encoder(32, 64, dilation=2)
        self.encoder4 = self._encoder(64, 128)
        self.encoder5 = self._encoder(128, 256, dilation=2)

        self.decoder5 = self._decoder(256 + 128, 128)
        self.decoder4 = self._decoder(128 + 64, 64)
        self.decoder3 = self._decoder(64 + 32, 32)
        self.decoder2 = self._decoder(32 + 16, 16)
        self.decoder1 = nn.Sequential(nn.Conv2d(16, 3, 3, padding=1), nn.Sigmoid())

    @staticmethod
    def _encoder(in_channels: int, out_channels: int, dilation: int = 1) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.ReLU(inplace=False),
            ResidualBlock(out_channels),
        )

    @staticmethod
    def _decoder(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=False),
            ResidualBlock(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        d5 = self.decoder5(torch.cat((e5, e4), dim=1))
        d4 = self.decoder4(torch.cat((d5, e3), dim=1))
        d3 = self.decoder3(torch.cat((d4, e2), dim=1))
        d2 = self.decoder2(torch.cat((d3, e1), dim=1))
        return self.decoder1(d2)


def _safe_torch_load(path: Path, device: torch.device) -> Any:
    """Load a trusted upstream checkpoint while preferring restricted unpickling."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # torch < 2.0 compatibility
        return torch.load(path, map_location=device)


def load_model(weights: str | Path, device: torch.device) -> CosmicClarityDenoiseCNN:
    weights_path = Path(weights)
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Model weights not found: {weights_path}. Run download_assets.py first."
        )
    checkpoint = _safe_torch_load(weights_path, device)
    if not isinstance(checkpoint, dict):
        raise ValueError("Unexpected checkpoint format: expected a mapping")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model = CosmicClarityDenoiseCNN().to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def model_statistics(model: nn.Module) -> dict[str, float | int]:
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "parameters": parameters,
        "trainable_parameters": trainable,
        "fp32_size_mib": parameters * 4 / 1024**2,
    }

