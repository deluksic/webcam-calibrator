"""Export the recommended ROI model to ONNX for the browser.

The exported graph accepts raw grayscale [0, 1] at 192x144, performs the same
per-image standardization used in training inside the graph, and returns a
full-resolution 192x144 mask probability. This keeps the browser preprocessing
to a simple grayscale resize.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from torch import nn

from blob_lab import RoiNetConfig, TinyRoiNet, load_roi_checkpoint

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = ROOT / "python" / "notebooks" / "blob_lab_data" / "roi_net_best.pt"
DEFAULT_OUTPUT = ROOT / "public" / "models" / "roi_tiny_dynamic.onnx"


NORMALIZATION_MEAN = 0.4
NORMALIZATION_STD = 0.2


class ExportRoiMask(nn.Module):
    def __init__(self, model: TinyRoiNet, config: RoiNetConfig):
        super().__init__()
        self.model = model
        self.height = config.input_height
        self.width = config.input_width

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # Fixed global normalization keeps the graph to simple elementwise ops
        # (no ReduceMean/Pow/Sqrt), which is friendlier to the WebGPU EP. It
        # also matched per-image normalization on the browser-resolution test.
        normalized = (image - NORMALIZATION_MEAN) / NORMALIZATION_STD
        logits, _presence = self.model(normalized)
        probabilities = torch.sigmoid(logits)
        return F.interpolate(
            probabilities,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    config = RoiNetConfig(
        channels=(3, 6, 12, 24),
        input_width=192,
        input_height=144,
        output_stride=2,
        false_negative_weight=10.0,
        tversky_weight=1.0,
    )
    model = load_roi_checkpoint(args.checkpoint, config, device="cpu")
    if model is None:
        raise SystemExit(f"Could not load {args.checkpoint}")
    model.eval()
    export_model = ExportRoiMask(model, config).eval()

    example = torch.rand(1, 1, config.input_height, config.input_width)
    with torch.no_grad():
        torch_output = export_model(example).numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        export_model,
        example,
        args.output,
        input_names=["input"],
        output_names=["mask"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
        dynamic_axes={
            "input": {2: "height", 3: "width"},
            "mask": {2: "height", 3: "width"},
        },
    )
    onnx.checker.check_model(str(args.output))

    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    onnx_output = session.run(["mask"], {"input": example.numpy()})[0]
    max_diff = float(np.max(np.abs(torch_output - onnx_output)))
    print(f"wrote {args.output} ({args.output.stat().st_size / 1024:.1f} KiB)")
    print(f"input {example.shape}, output {onnx_output.shape}, max_diff={max_diff:.3e}")
    if max_diff > 1e-4:
        raise SystemExit("ONNX output does not match PyTorch")
    # Verify the dynamic height path used by the browser.
    for height, width in ((108, 192), (144, 192), (192, 192)):
        probe = torch.rand(1, 1, height, width)
        with torch.no_grad():
            expected = export_model(probe).numpy()
        actual = session.run(["mask"], {"input": probe.numpy()})[0]
        probe_diff = float(np.max(np.abs(expected - actual)))
        print(f"dynamic probe {height}x{width}: {actual.shape}, max_diff={probe_diff:.3e}")
        if probe_diff > 1e-4:
            raise SystemExit("Dynamic ONNX probe failed")
    (args.output.with_suffix(".json")).write_text(
        json.dumps(
            {
                "input": [1, 1, "height", "width"],
                "output": [1, 1, "height", "width"],
                "normalization": f"(x - {NORMALIZATION_MEAN}) / {NORMALIZATION_STD} inside graph",
                "mask_threshold": 0.5,
                "inference_margin_fraction": 0.03,
                "checkpoint": str(args.checkpoint),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
