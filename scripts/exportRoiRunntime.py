"""Export the ROI net for the runntime spike.

Writes three files into public/spike/:
  models/roi_net.safetensors   weights with runntime-friendly names (PyTorch `net.N`
                        slots renamed to conv1/bn1/conv2/bn2, BN left unfused
                        so runntime's foldBnIntoConv can do it lazily)
  input.f32.bin         one real frame as z-scored CHW f32, [1, H, W]
  ref_mask.f32.bin      PyTorch f32 sigmoid output for that input, [H/2, W/2]

The input and reference let the browser compare runntime against PyTorch on
identical bytes, with no graph runtime in the loop.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from safetensors.numpy import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python" / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from blob_lab import load_roi_checkpoint  # noqa: E402
from fp16RoiCheck import luma_bilinear, normalize  # noqa: E402

CKPT = ROOT / "python" / "notebooks" / "blob_lab_data" / "roi_net_best.pt"
SAMPLE = ROOT / "python" / "notebooks" / "blob_lab_data" / "roi_dataset_preview" / "sample_002.png"
OUT = ROOT / "public" / "spike"
# The model loads this exact file; public/spike/ only holds the spike's test tensors.
WEIGHTS = ROOT / "public" / "models" / "roi_net.safetensors"

H, W = 192, 256

# ConvBlock layout: net = [conv, bn, relu, conv, bn, relu]. Each (conv, bn)
# pair is emitted as one folded conv so the file can be f16 without rounding
# BatchNorm's running stats through half precision first.
BLOCKS = ("enc1", "enc2", "enc3", "enc4", "bottleneck", "dec3", "dec2")
PAIRS = (("conv1", 0, 1), ("conv2", 3, 4))
BN_EPS = 1e-5  # torch's BatchNorm2d default, and runntime's foldBnIntoConv default


def fold_bn_into_conv(
    conv: nn.Conv2d, bn: nn.BatchNorm2d
) -> tuple[np.ndarray, np.ndarray]:
    """Return (weight, bias) with the BN affine folded in, in f32.

    Mirrors runntime's foldBnIntoConv: per out-channel
    scale = gamma / sqrt(var + eps), weight' = weight * scale,
    bias' = beta + (bias - mean) * scale.
    """
    weight = conv.weight.detach().cpu().numpy().astype(np.float64)
    if conv.bias is not None:
        bias = conv.bias.detach().cpu().numpy().astype(np.float64)
    else:
        bias = np.zeros(weight.shape[0], dtype=np.float64)
    gamma = bn.weight.detach().cpu().numpy().astype(np.float64)
    beta = bn.bias.detach().cpu().numpy().astype(np.float64)
    mean = bn.running_mean.detach().cpu().numpy().astype(np.float64)
    var = bn.running_var.detach().cpu().numpy().astype(np.float64)
    scale = gamma / np.sqrt(var + BN_EPS)
    folded_weight = weight * scale[:, None, None, None]
    folded_bias = beta + (bias - mean) * scale
    out_c = folded_weight.shape[0]
    return (
        folded_weight.reshape(out_c, -1).astype(np.float32),
        folded_bias.astype(np.float32),
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = load_roi_checkpoint(CKPT, None, device="cpu")
    if model is None:
        raise SystemExit(f"could not load {CKPT}")
    model.eval()

    # BN folds in f32 here, then the folded conv is stored as f16 — exactly the
    # arithmetic runntime would have done at load, so the deployed numbers are
    # unchanged, but the file is half the size and BatchNorm stats are never
    # rounded to half precision. Omitting the bn.* keys also means the runtime
    # fold becomes a no-op.
    tensors: dict[str, np.ndarray] = {}
    for block in BLOCKS:
        module = getattr(model, block)
        for name, conv_index, bn_index in PAIRS:
            weight, bias = fold_bn_into_conv(
                module.net[conv_index], module.net[bn_index]
            )
            tensors[f"{block}.{name}.weight"] = weight.astype(np.float16)
            tensors[f"{block}.{name}.bias"] = bias.astype(np.float16)

    # The mask head has no BatchNorm.
    head = model.mask_head
    tensors["mask_head.weight"] = (
        head.weight.detach().cpu().numpy().reshape(head.out_channels, -1).astype(np.float16)
    )
    tensors["mask_head.bias"] = head.bias.detach().cpu().numpy().astype(np.float16)

    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, WEIGHTS)
    print(f"weights: {len(tensors)} f16 tensors, "
          f"{sum(t.nbytes for t in tensors.values()) / 1024:.1f} KiB -> {WEIGHTS}")

    # One real frame, preprocessed exactly like the GPU prepare pass. Several
    # preview samples are negatives (empty mask), which would make the
    # comparison vacuous, so pick the frame with the most predicted content.
    from PIL import Image

    best = None
    for path in sorted(SAMPLE.parent.glob("sample_*.png")):
        if "_mask" in path.name:
            continue
        candidate = normalize(luma_bilinear(np.asarray(Image.open(path).convert("RGB")), W, H)[None])
        with torch.no_grad():
            probe = torch.sigmoid(model(torch.from_numpy(candidate[None].astype(np.float32)))[0])
        score = float((probe > 0.5).float().mean())
        if best is None or score > best[0]:
            best = (score, path, candidate)
    if best is None:
        raise SystemExit("no samples found")
    score, picked, chw = best
    print(f"picked {picked.name} (mask coverage {score:.3f})")
    chw = chw.astype(np.float32)  # [1, H, W]
    (OUT / "input.f32.bin").write_bytes(chw.tobytes())

    with torch.no_grad():
        logits, _presence = model(torch.from_numpy(chw[None]))  # [1,1,H,W]
        ref = torch.sigmoid(logits)[0, 0].numpy().astype(np.float32)
    (OUT / "ref_mask.f32.bin").write_bytes(ref.tobytes())

    print(f"input {chw.shape} -> {OUT / 'input.f32.bin'}")
    print(f"reference mask {ref.shape} mean={ref.mean():.4f} max={ref.max():.4f}")


if __name__ == "__main__":
    main()
