"""Export a WebGPU-friendly native-resolution ROI mask model.

Input is already per-image standardized (z = (luma - mean) / std). Output is the
native decoder mask at half the input resolution. There is no normalization,
Resize, Shape, or Slice in the graph - only Conv/ReLU/MaxPool/Concat/Sigmoid.
The browser upsamples the mask texture for display.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from blob_lab import RoiNetConfig, TinyRoiNet, load_roi_checkpoint

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "python" / "notebooks" / "blob_lab_data" / "roi_net_best.pt"
OUTPUT = ROOT / "public" / "models" / "roi_tiny_native.onnx"


class NativeMask(nn.Module):
    def __init__(self, model: TinyRoiNet):
        super().__init__()
        self.model = model

    def forward(self, normalized: torch.Tensor) -> torch.Tensor:
        logits, _presence = self.model(normalized)
        return torch.sigmoid(logits)


def main() -> None:
    config = RoiNetConfig(
        channels=(3, 6, 12, 24),
        input_width=192,
        input_height=144,
        output_stride=2,
        false_negative_weight=10.0,
        tversky_weight=1.0,
    )
    model = load_roi_checkpoint(CHECKPOINT, config, device="cpu")
    if model is None:
        raise SystemExit(f"Could not load {CHECKPOINT}")
    export_model = NativeMask(model.eval()).eval()
    example = torch.rand(1, 1, 144, 192)
    with torch.no_grad():
        expected = export_model(example).numpy()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        export_model,
        example,
        OUTPUT,
        input_names=["input"],
        output_names=["mask"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
        dynamic_axes={
            "input": {2: "height", 3: "width"},
            "mask": {2: "mask_height", 3: "mask_width"},
        },
    )
    onnx.checker.check_model(str(OUTPUT))
    session = ort.InferenceSession(str(OUTPUT), providers=["CPUExecutionProvider"])
    for height, width in ((144, 192), (108, 192), (192, 192)):
        probe = torch.rand(1, 1, height, width)
        with torch.no_grad():
            torch_out = export_model(probe).numpy()
        onnx_out = session.run(["mask"], {"input": probe.numpy()})[0]
        diff = float(np.max(np.abs(torch_out - onnx_out)))
        print(f"{height}x{width} -> {onnx_out.shape}, max_diff={diff:.3e}")
        if diff > 1e-4:
            raise SystemExit("Native ONNX probe failed")
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size / 1024:.1f} KiB)")
    (OUTPUT.with_suffix(".json")).write_text(
        json.dumps(
            {
                "input": [1, 1, "height", "width"],
                "output": [1, 1, "height/2", "width/2"],
                "normalization": "per-image z computed in browser",
                "mask_threshold": 0.5,
                "inference_margin_fraction": 0.03,
                "checkpoint": str(CHECKPOINT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
