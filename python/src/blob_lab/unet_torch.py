"""Tiny U-Net heatmap detector using PyTorch/MPS on Apple Silicon."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class UNetTorchConfig:
    filters: int = 8
    input_channels: int = 4
    epochs: int = 25
    batch_size: int = 1
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    focal_gamma: float = 2.0
    center_weight: float = 1.0
    area_weight: float = 1.0
    junction_weight: float = 0.5
    seed: int = 0
    device: str | None = None


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyUNet(nn.Module):
    def __init__(self, filters: int = 5, input_channels: int = 4):
        super().__init__()
        self.input_channels = int(input_channels)
        self.enc1 = ConvBlock(self.input_channels, filters)
        self.enc2 = ConvBlock(filters, 2 * filters)
        self.enc3 = ConvBlock(2 * filters, 4 * filters)
        # Extra depth at 1/8 resolution; context comes from pooling, not dilation.
        self.bottleneck = ConvBlock(4 * filters, 8 * filters)
        self.dec3 = ConvBlock(8 * filters + 4 * filters, 4 * filters)
        self.dec2 = ConvBlock(4 * filters + 2 * filters, 2 * filters)
        self.dec1 = ConvBlock(2 * filters + filters, filters)
        self.out = nn.Conv2d(filters, 1, 1)
        self.area_head = nn.Conv2d(filters, 1, 1)
        self.junction_head = nn.Conv2d(filters, 1, 1)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        b = self.bottleneck(F.max_pool2d(e3, 2))
        d3 = F.interpolate(b, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        heatmap = self.out(d1)
        if return_aux:
            return heatmap, self.area_head(d1), self.junction_head(d1)
        return heatmap


def resolve_device(preference: str | None = None) -> torch.device:
    if preference is not None:
        return torch.device(preference)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class TrainedUNetTorch:
    model: TinyUNet
    history: dict
    config: UNetTorchConfig
    device: torch.device

    def score(self, images: np.ndarray) -> np.ndarray:
        return predict_heatmap(self.model, images, self.device)

    def score_heads(self, images: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return predict_heads(self.model, images, self.device)


def _to_nchw(images: np.ndarray) -> np.ndarray:
    return np.transpose(images, (0, 3, 1, 2)).astype(np.float32)


def heatmap_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    config: UNetTorchConfig,
) -> torch.Tensor:
    positive = targets.sum()
    negative = targets.numel() - positive
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    pt = targets * probs + (1.0 - targets) * (1.0 - probs)
    focal = (1.0 - pt).pow(config.focal_gamma) * bce
    # Average each class separately.  A pixel-weighted mean would still drown
    # a few hundred positive pixels in millions of background pixels.
    positive_loss = (focal * targets).sum() / (positive + 1.0)
    negative_loss = (focal * (1.0 - targets)).sum() / (negative + 1.0)
    return positive_loss + negative_loss


def _area_loss(area_logits, area_targets) -> torch.Tensor:
    target = F.interpolate(area_targets, size=area_logits.shape[-2:], mode="area")
    bce = F.binary_cross_entropy_with_logits(area_logits, target)
    probs = torch.sigmoid(area_logits)
    dice = 1.0 - (2.0 * (probs * target).sum() + 1.0) / (probs.sum() + target.sum() + 1.0)
    return bce + 0.5 * dice


def train_unet_torch(
    train_images: np.ndarray,
    train_targets: np.ndarray,
    validation_images: np.ndarray,
    validation_targets: np.ndarray,
    config: UNetTorchConfig | None = None,
    on_epoch=None,
    initial_model: TinyUNet | None = None,
    train_features: np.ndarray | None = None,
    validation_features: np.ndarray | None = None,
    train_masks: np.ndarray | None = None,
    validation_masks: np.ndarray | None = None,
    train_junctions: np.ndarray | None = None,
    validation_junctions: np.ndarray | None = None,
) -> TrainedUNetTorch:
    config = config or UNetTorchConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = resolve_device(config.device)
    model = (
        initial_model.to(device)
        if initial_model is not None
        else TinyUNet(config.filters, config.input_channels).to(device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    train_inputs = train_features if train_features is not None else train_images
    validation_inputs = validation_features if validation_features is not None else validation_images
    train_x = torch.from_numpy(_to_nchw(train_inputs)).to(device)
    train_y = torch.from_numpy(np.transpose(train_targets, (0, 3, 1, 2)).astype(np.float32)).to(device)
    val_x = torch.from_numpy(_to_nchw(validation_inputs)).to(device)
    val_y = torch.from_numpy(np.transpose(validation_targets, (0, 3, 1, 2)).astype(np.float32)).to(device)
    train_mask_t = torch.from_numpy(np.transpose(train_masks, (0, 3, 1, 2)).astype(np.float32)).to(device) if train_masks is not None else None
    val_mask_t = torch.from_numpy(np.transpose(validation_masks, (0, 3, 1, 2)).astype(np.float32)).to(device) if validation_masks is not None else None
    train_junction_t = torch.from_numpy(np.transpose(train_junctions, (0, 3, 1, 2)).astype(np.float32)).to(device) if train_junctions is not None else None
    val_junction_t = torch.from_numpy(np.transpose(validation_junctions, (0, 3, 1, 2)).astype(np.float32)).to(device) if validation_junctions is not None else None

    history = {"train_loss": [], "val_loss": [], "device": str(device)}
    for epoch in range(config.epochs):
        model.train()
        order = torch.randperm(len(train_x), device=device)
        losses = []
        for start in range(0, len(order), config.batch_size):
            batch = order[start : start + config.batch_size]
            heat_logits, area_logits, junction_logits = model(train_x[batch], return_aux=True)
            loss = config.center_weight * heatmap_loss(heat_logits, train_y[batch], config)
            if train_mask_t is not None:
                loss = loss + config.area_weight * _area_loss(
                    area_logits, train_mask_t[batch]
                )
            if train_junction_t is not None:
                loss = loss + config.junction_weight * heatmap_loss(
                    junction_logits, train_junction_t[batch], config
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            heat_logits, area_logits, junction_logits = model(val_x, return_aux=True)
            val_loss = float((config.center_weight * heatmap_loss(heat_logits, val_y, config)).detach().cpu())
            if val_mask_t is not None:
                val_loss += float(
                    (config.area_weight * _area_loss(area_logits, val_mask_t)).detach().cpu()
                )
            if val_junction_t is not None:
                val_loss += float(
                    (config.junction_weight * heatmap_loss(junction_logits, val_junction_t, config)).detach().cpu()
                )
        train_loss = float(np.mean(losses)) if losses else 0.0
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if on_epoch is not None:
            on_epoch(epoch + 1, config.epochs, train_loss, val_loss)
    model.eval()
    return TrainedUNetTorch(model, history, config, device)


@torch.no_grad()
def predict_heatmap(model: TinyUNet, images: np.ndarray, device: torch.device | None = None) -> np.ndarray:
    device = device or next(model.parameters()).device
    model.eval()
    outputs = []
    for start in range(0, len(images), 4):
        batch = torch.from_numpy(_to_nchw(images[start : start + 4])).to(device)
        probabilities = torch.sigmoid(model(batch))
        outputs.append(probabilities[:, 0].cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32)


@torch.no_grad()
def predict_heads(
    model: TinyUNet,
    images: np.ndarray,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = device or next(model.parameters()).device
    model.eval()
    heatmaps, areas, junctions = [], [], []
    for start in range(0, len(images), 4):
        batch = torch.from_numpy(_to_nchw(images[start : start + 4])).to(device)
        heat, area, junction = model(batch, return_aux=True)
        height, width = batch.shape[-2:]
        areas.append(
            torch.sigmoid(
                F.interpolate(area, size=(height, width), mode="bilinear", align_corners=False)
            )[:, 0]
            .cpu()
            .numpy()
        )
        heatmaps.append(torch.sigmoid(heat)[:, 0].cpu().numpy())
        junctions.append(torch.sigmoid(junction)[:, 0].cpu().numpy())
    return (
        np.concatenate(heatmaps, axis=0).astype(np.float32),
        np.concatenate(areas, axis=0).astype(np.float32),
        np.concatenate(junctions, axis=0).astype(np.float32),
    )


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))



def save_unet_checkpoint(
    path: str | Path,
    model: TinyUNet,
    filters: int,
    input_channels: int | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "filters": int(filters),
            "input_channels": int(input_channels or model.input_channels),
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_unet_checkpoint(
    path: str | Path,
    filters: int,
    input_channels: int = 4,
    device: str | None = None,
) -> TinyUNet | None:
    path = Path(path)
    if not path.is_file():
        return None
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if int(checkpoint.get("filters", -1)) != int(filters):
        return None
    if int(checkpoint.get("input_channels", -1)) != int(input_channels):
        return None
    model = TinyUNet(int(filters), int(input_channels))
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except RuntimeError:
        return None
    return model.to(resolve_device(device))
