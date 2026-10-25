"""Low-resolution target-area (ROI) segmentation for the hex target.

The model is intentionally tiny: it runs on a downscaled grayscale frame and
predicts the general target area plus a frame-level presence score. The naive
full-resolution peak finder then runs inside the predicted mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from blob_lab.metrics import roc_auc
from blob_lab.unet_torch import resolve_device


@dataclass(frozen=True)
class RoiNetConfig:
    channels: tuple[int, int, int, int] = (8, 16, 32, 64)
    input_width: int = 192
    input_height: int = 144
    output_stride: int = 2
    presence_weight: float = 0.2
    # Missing target pixels removes usable markers, so weight them much more
    # heavily than background false positives.
    false_negative_weight: float = 5.0
    tversky_weight: float = 0.5
    tversky_beta: float = 0.7
    epochs: int = 20
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 0
    device: str | None = None

    @property
    def mask_height(self) -> int:
        return max(1, self.input_height // self.output_stride)

    @property
    def mask_width(self) -> int:
        return max(1, self.input_width // self.output_stride)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyRoiNet(nn.Module):
    """Four-level U-Net with a mask head and a global presence head."""

    def __init__(self, config: RoiNetConfig = RoiNetConfig()):
        super().__init__()
        if config.output_stride not in (1, 2):
            raise ValueError("output_stride must be 1 or 2")
        self.config = config
        c1, c2, c3, c4 = config.channels
        self.enc1 = ConvBlock(1, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.enc4 = ConvBlock(c3, c4)
        self.bottleneck = ConvBlock(c4, c4)
        self.dec3 = ConvBlock(c4 + c3, c3)
        self.dec2 = ConvBlock(c3 + c2, c2)
        self.dec1 = ConvBlock(c2 + c1, c1) if config.output_stride == 1 else None
        out_channels = c1 if config.output_stride == 1 else c2
        self.mask_head = nn.Conv2d(out_channels, 1, 1)
        self.presence_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c4, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        b = self.bottleneck(F.max_pool2d(e4, 2))
        d3 = self.dec3(
            torch.cat(
                [
                    F.interpolate(b, size=e3.shape[-2:], mode="bilinear", align_corners=False),
                    e3,
                ],
                dim=1,
            )
        )
        d2 = self.dec2(
            torch.cat(
                [
                    F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False),
                    e2,
                ],
                dim=1,
            )
        )
        if self.dec1 is not None:
            d2 = self.dec1(
                torch.cat(
                    [
                        F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False),
                        e1,
                    ],
                    dim=1,
                )
            )
        return self.mask_head(d2), self.presence_head(b)


def parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def prepare_roi_input(
    images: np.ndarray,
    config: RoiNetConfig,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Resize to the ROI input size and standardize each frame."""
    array = np.asarray(images, dtype=np.float32)
    if array.ndim == 4:
        array = array[..., 0]
    if array.ndim == 2:
        array = array[None]
    if array.ndim != 3:
        raise ValueError(f"Expected (N,H,W) or (N,H,W,1), got {array.shape}")
    resized = np.stack(
        [
            cv2.resize(
                image,
                (config.input_width, config.input_height),
                interpolation=cv2.INTER_AREA,
            )
            for image in array
        ],
        axis=0,
    )
    mean = resized.mean(axis=(1, 2), keepdims=True)
    std = resized.std(axis=(1, 2), keepdims=True)
    normalized = (resized - mean) / np.maximum(std, 1e-5)
    tensor = torch.from_numpy(normalized[:, None].astype(np.float32))
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def prepare_roi_targets(
    masks: np.ndarray,
    config: RoiNetConfig,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    array = np.asarray(masks, dtype=np.float32)
    if array.ndim == 4:
        array = array[..., 0]
    resized = np.stack(
        [
            cv2.resize(
                mask,
                (config.mask_width, config.mask_height),
                interpolation=cv2.INTER_AREA,
            )
            for mask in array
        ],
        axis=0,
    )
    tensor = torch.from_numpy(np.clip(resized[:, None], 0.0, 1.0))
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def mask_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    false_negative_weight: float = 1.0,
    tversky_weight: float = 0.0,
    tversky_beta: float = 0.7,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    if false_negative_weight != 1.0:
        weights = 1.0 + (float(false_negative_weight) - 1.0) * targets
        bce = (
            F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            * weights
        ).mean()
    else:
        bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = 1.0 - (2.0 * (probabilities * targets).sum() + 1.0) / (
        probabilities.sum() + targets.sum() + 1.0
    )
    loss = bce + dice
    if tversky_weight > 0.0:
        beta = float(tversky_beta)
        alpha = 1.0 - beta
        true_positive = (probabilities * targets).sum()
        false_positive = (probabilities * (1.0 - targets)).sum()
        false_negative = ((1.0 - probabilities) * targets).sum()
        tversky = 1.0 - (true_positive + 1.0) / (
            true_positive
            + alpha * false_positive
            + beta * false_negative
            + 1.0
        )
        loss = loss + float(tversky_weight) * tversky
    return loss


@dataclass
class TrainedRoiNet:
    model: TinyRoiNet
    history: dict[str, Any]
    config: RoiNetConfig
    device: torch.device

    def predict(self, images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return predict_roi(self.model, images, self.config, self.device)


def train_roi(
    train_images: np.ndarray,
    train_masks: np.ndarray,
    train_presence: np.ndarray,
    validation_images: np.ndarray,
    validation_masks: np.ndarray,
    validation_presence: np.ndarray,
    config: RoiNetConfig | None = None,
    on_epoch=None,
    initial_model: TinyRoiNet | None = None,
) -> TrainedRoiNet:
    config = config or RoiNetConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = resolve_device(config.device)
    model = (
        initial_model.to(device)
        if initial_model is not None
        else TinyRoiNet(config).to(device)
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    train_x = prepare_roi_input(train_images, config, device)
    train_y = prepare_roi_targets(train_masks, config, device)
    train_p = torch.from_numpy(np.asarray(train_presence, dtype=np.float32)[:, None]).to(device)
    val_x = prepare_roi_input(validation_images, config, device)
    val_y = prepare_roi_targets(validation_masks, config, device)
    val_p = torch.from_numpy(
        np.asarray(validation_presence, dtype=np.float32)[:, None]
    ).to(device)

    history: dict[str, Any] = {
        "train_loss": [],
        "val_loss": [],
        "train_mask_loss": [],
        "val_mask_loss": [],
        "device": str(device),
    }
    for epoch in range(config.epochs):
        model.train()
        order = torch.randperm(len(train_x), device=device)
        losses: list[float] = []
        mask_losses: list[float] = []
        for start in range(0, len(order), config.batch_size):
            batch = order[start : start + config.batch_size]
            logits, presence_logits = model(train_x[batch])
            mask_term = mask_loss(
                logits,
                train_y[batch],
                false_negative_weight=config.false_negative_weight,
                tversky_weight=config.tversky_weight,
                tversky_beta=config.tversky_beta,
            )
            presence_term = F.binary_cross_entropy_with_logits(
                presence_logits, train_p[batch]
            )
            loss = mask_term + config.presence_weight * presence_term
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            mask_losses.append(float(mask_term.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_logits, val_presence_logits = model(val_x)
            val_mask_term = mask_loss(
                val_logits,
                val_y,
                false_negative_weight=config.false_negative_weight,
                tversky_weight=config.tversky_weight,
                tversky_beta=config.tversky_beta,
            )
            val_loss = val_mask_term + config.presence_weight * F.binary_cross_entropy_with_logits(
                val_presence_logits, val_p
            )
        history["train_loss"].append(float(np.mean(losses)) if losses else 0.0)
        history["val_loss"].append(float(val_loss.detach().cpu()))
        history["train_mask_loss"].append(
            float(np.mean(mask_losses)) if mask_losses else 0.0
        )
        history["val_mask_loss"].append(float(val_mask_term.detach().cpu()))
        if on_epoch is not None:
            on_epoch(
                epoch + 1,
                config.epochs,
                history["train_loss"][-1],
                history["val_loss"][-1],
            )
    model.eval()
    return TrainedRoiNet(model, history, config, device)


@torch.no_grad()
def predict_roi(
    model: TinyRoiNet,
    images: np.ndarray,
    config: RoiNetConfig | None = None,
    device: torch.device | str | None = None,
    *,
    upsample: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mask probability, presence probability) at the ROI input size."""
    config = config or model.config
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    inputs = prepare_roi_input(images, config, device)
    logits, presence_logits = model(inputs)
    probabilities = torch.sigmoid(logits)
    if upsample:
        probabilities = F.interpolate(
            probabilities,
            size=(config.input_height, config.input_width),
            mode="bilinear",
            align_corners=False,
        )
    return (
        probabilities[:, 0].cpu().numpy().astype(np.float32),
        torch.sigmoid(presence_logits)[:, 0].cpu().numpy().astype(np.float32),
    )


def resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(
        np.asarray(mask, dtype=np.float32), (int(width), int(height)), interpolation=cv2.INTER_LINEAR
    )


def largest_roi_polygon(
    mask: np.ndarray,
    *,
    threshold: float = 0.5,
    margin_px: int = 0,
) -> np.ndarray:
    """Return the convex hull of the largest predicted region, or an empty array."""
    binary = (np.asarray(mask, dtype=np.float32) >= threshold).astype(np.uint8)
    if margin_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * int(margin_px) + 1, 2 * int(margin_px) + 1)
        )
        binary = cv2.dilate(binary, kernel, iterations=1)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((0, 2), dtype=np.float32)
    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour).reshape(-1, 2)
    return hull.astype(np.float32)


def roi_mask_from_probability(
    probability: np.ndarray,
    width: int,
    height: int,
    *,
    threshold: float = 0.5,
    margin_fraction: float = 0.03,
) -> np.ndarray:
    """Convert a low-res ROI probability into an inclusive full-res mask.

    The largest predicted region is reduced to its convex hull, scaled to the
    output resolution, and dilated so the peak finder does not clip markers at
    the boundary.
    """
    probability = np.asarray(probability, dtype=np.float32)
    if probability.ndim != 2:
        raise ValueError(f"Expected a 2D probability map, got {probability.shape}")
    low_height, low_width = probability.shape
    binary = (probability >= threshold).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros((int(height), int(width)), dtype=np.float32)
    if not contours:
        return mask
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) <= 0.0:
        return mask
    hull = cv2.convexHull(contour).reshape(-1, 2).astype(np.float32)
    hull[:, 0] *= float(width) / max(1.0, float(low_width))
    hull[:, 1] *= float(height) / max(1.0, float(low_height))
    cv2.fillConvexPoly(mask, np.round(hull).astype(np.int32), 1.0)
    margin_px = int(round(float(margin_fraction) * min(width, height)))
    if margin_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * margin_px + 1, 2 * margin_px + 1)
        )
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask.astype(np.float32)


def evaluate_roi(
    mask_probabilities: np.ndarray,
    presence_scores: np.ndarray,
    dataset,
    *,
    threshold: float = 0.5,
    presence_threshold: float = 0.5,
) -> dict[str, float]:
    """Mask IoU/Dice, presence metrics, and downstream center containment."""
    height, width = dataset.images.shape[1:3]
    predicted = np.stack(
        [resize_mask(mask, width, height) >= threshold for mask in mask_probabilities],
        axis=0,
    ).astype(np.float32)
    truth = (np.asarray(dataset.masks[..., 0]) >= 0.5).astype(np.float32)
    intersection = (predicted * truth).sum(axis=(1, 2))
    union = np.clip(predicted + truth - predicted * truth, 0.0, 1.0).sum(axis=(1, 2))
    per_image_iou = np.where(union > 0.0, intersection / np.maximum(union, 1.0), 1.0)
    iou = float(np.mean(per_image_iou))
    positive_images = truth.sum(axis=(1, 2)) > 0
    negative_images = ~positive_images
    true_positive = (predicted * truth).sum(axis=(1, 2))
    false_negative = ((1.0 - predicted) * truth).sum(axis=(1, 2))
    false_positive = (predicted * (1.0 - truth)).sum(axis=(1, 2))
    true_negative = ((1.0 - predicted) * (1.0 - truth)).sum(axis=(1, 2))
    false_negative_rate = false_negative / np.maximum(true_positive + false_negative, 1.0)
    false_positive_rate = false_positive / np.maximum(false_positive + true_negative, 1.0)
    dice = float(
        np.mean((2.0 * intersection + 1.0) / (predicted.sum(axis=(1, 2)) + truth.sum(axis=(1, 2)) + 1.0))
    )
    presence = (np.asarray(presence_scores) >= presence_threshold).astype(np.float32)
    presence_accuracy = float(np.mean(presence == np.asarray(dataset.presence)))
    try:
        auc = float(roc_auc(np.asarray(dataset.presence), np.asarray(presence_scores)))
    except Exception:
        auc = 0.5

    containments: list[float] = []
    area_ratios: list[float] = []
    for index, meta in enumerate(dataset.metadata):
        if not meta.get("points"):
            continue
        mask = predicted[index] > 0.5
        contained = 0
        for x, y in meta["points"]:
            ix = int(np.clip(round(x), 0, width - 1))
            iy = int(np.clip(round(y), 0, height - 1))
            contained += int(mask[iy, ix])
        containments.append(contained / max(1, len(meta["points"])))
        truth_area = float(truth[index].sum())
        if truth_area > 0:
            area_ratios.append(float(mask.sum()) / truth_area)
    return {
        "mask_iou": iou,
        "mask_dice": dice,
        "positive_mask_iou": float(np.mean(per_image_iou[positive_images]))
        if positive_images.any()
        else 0.0,
        "positive_count": int(positive_images.sum()),
        "negative_count": int(negative_images.sum()),
        "negative_mask_area": float(predicted[negative_images].mean())
        if negative_images.any()
        else 0.0,
        "false_negative_rate": float(np.mean(false_negative_rate[positive_images]))
        if positive_images.any()
        else 0.0,
        "false_positive_rate": float(np.mean(false_positive_rate)),
        "center_miss_rate": 1.0
        - (float(np.mean(containments)) if containments else 0.0),
        "presence_accuracy": presence_accuracy,
        "presence_auc": auc,
        "center_containment": float(np.mean(containments)) if containments else 0.0,
        "area_ratio": float(np.mean(area_ratios)) if area_ratios else 0.0,
    }


def evaluate_roi_groups(
    mask_probabilities: np.ndarray,
    presence_scores: np.ndarray,
    dataset,
    *,
    groups: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, dict[str, float]]:
    """Evaluate the same predictions split by dataset branch.

    The mixed metrics hide the scale-specific behaviour: a model can score
    well overall while missing full-view targets entirely. These groups make
    the close-up/far-target tradeoff explicit.
    """
    if groups is None:
        groups = {
            "positive": lambda meta: float(meta.get("presence", 1)) >= 0.5,
            "full_view": lambda meta: bool(meta.get("full_view", False)),
            "closeup": lambda meta: bool(meta.get("closeup", False)),
            "cropped": lambda meta: bool(meta.get("cropped", False)),
            "regular": lambda meta: float(meta.get("presence", 1)) >= 0.5
            and not meta.get("full_view", False)
            and not meta.get("closeup", False)
            and not meta.get("cropped", False),
            "heavy_blur": lambda meta: bool(meta.get("heavy_blur", False)),
            "negative": lambda meta: float(meta.get("presence", 1)) < 0.5,
        }
    probabilities = np.asarray(mask_probabilities)
    presence = np.asarray(presence_scores)
    results: dict[str, dict[str, float]] = {}
    for name, predicate in groups.items():
        indices = np.asarray(
            [
                index
                for index, meta in enumerate(dataset.metadata)
                if predicate(meta)
            ],
            dtype=np.int64,
        )
        if indices.size == 0:
            continue
        metrics = evaluate_roi(
            probabilities[indices], presence[indices], dataset.subset(indices), **kwargs
        )
        metrics["samples"] = int(indices.size)
        results[name] = metrics
    return results


def save_roi_checkpoint(path: str | Path, model: TinyRoiNet, config: RoiNetConfig) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "channels": tuple(config.channels),
            "input_width": int(config.input_width),
            "input_height": int(config.input_height),
            "output_stride": int(config.output_stride),
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_roi_checkpoint(
    path: str | Path,
    config: RoiNetConfig | None = None,
    device: str | None = None,
) -> TinyRoiNet | None:
    path = Path(path)
    if not path.is_file():
        return None
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if config is None:
        config = RoiNetConfig(
            channels=tuple(checkpoint.get("channels", (8, 16, 32, 64))),
            input_width=int(checkpoint.get("input_width", 192)),
            input_height=int(checkpoint.get("input_height", 144)),
            output_stride=int(checkpoint.get("output_stride", 2)),
        )
    model = TinyRoiNet(config)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except RuntimeError:
        return None
    return model.to(resolve_device(device))
