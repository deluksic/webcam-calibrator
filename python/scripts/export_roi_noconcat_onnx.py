"""Export the ROI U-Net with additive skip fusion instead of Concat.

ORT WebGPU mis-executes Concat in this environment. A 3x3 convolution over
``cat([upsampled, skip])`` is exactly ``conv(upsampled) + conv(skip)`` because
convolution is linear in its input channels, so the decoder weights can be
split and the concatenation removed without retraining.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from torch import nn

from blob_lab import RoiNetConfig, load_roi_checkpoint
from blob_lab.roi_net import ConvBlock

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "python" / "notebooks" / "blob_lab_data" / "roi_net_best.pt"
OUTPUT = ROOT / "public" / "models" / "roi_tiny_no_concat.onnx"


class PadChannels(nn.Module):
    """Pad a 6-channel feature map to 8 channels with zeros."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, (0, 0, 0, 0, 0, 2))


def _expand_conv_input(conv: nn.Conv2d, target_channels: int = 8) -> nn.Conv2d:
    expanded = nn.Conv2d(
        target_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        bias=conv.bias is not None,
    )
    with torch.no_grad():
        expanded.weight.zero_()
        expanded.weight[:, : conv.in_channels].copy_(conv.weight)
        if conv.bias is not None:
            expanded.bias.copy_(conv.bias)
    return expanded


class PaddedConvBlock(nn.Module):
    def __init__(self, block: ConvBlock):
        super().__init__()
        self.block = block
        if block.net[0].in_channels == 6:
            conv = _expand_conv_input(block.net[0])
            self.pad = PadChannels()
            self.block.net[0] = conv
        else:
            self.pad = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad is not None:
            x = self.pad(x)
        return self.block(x)


class SplitConvAdd(nn.Module):
    def __init__(self, conv: nn.Conv2d, first_channels: int):
        super().__init__()
        self.conv_a = nn.Conv2d(
            first_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            bias=conv.bias is not None,
        )
        second_channels = conv.in_channels - first_channels
        self.conv_b = nn.Conv2d(
            second_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            bias=conv.bias is not None,
        )
        with torch.no_grad():
            self.conv_a.weight.copy_(conv.weight[:, :first_channels])
            self.conv_b.weight.copy_(conv.weight[:, first_channels:])
            if conv.bias is not None:
                # Split the bias once; the other branch gets zero.
                self.conv_a.bias.copy_(conv.bias)
                self.conv_b.bias.zero_()
        self.pad_second = None
        if second_channels == 6:
            self.pad_second = PadChannels()
            with torch.no_grad():
                self.conv_b = _expand_conv_input(self.conv_b)

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        if self.pad_second is not None:
            second = self.pad_second(second)
        return self.conv_a(first) + self.conv_b(second)


class AddSkipBlock(nn.Module):
    def __init__(self, block: ConvBlock, first_channels: int):
        super().__init__()
        self.first = SplitConvAdd(block.net[0], first_channels)
        self.rest = nn.Sequential(*list(block.net)[1:])

    def forward(self, upsampled: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.rest(self.first(upsampled, skip))


class NoConcatRoi(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.m = model
        c1, c2, c3, c4 = model.config.channels
        self.enc3 = PaddedConvBlock(model.enc3)
        self.dec3 = AddSkipBlock(model.dec3, c4)
        self.dec2 = AddSkipBlock(model.dec2, c3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.m.enc1(x)
        e2 = self.m.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.m.enc4(F.max_pool2d(e3, 2))
        b = self.m.bottleneck(F.max_pool2d(e4, 2))
        d3 = self.dec3(F.interpolate(b, scale_factor=4, mode="bilinear", align_corners=False), e3)
        d2 = self.dec2(F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=False), e2)
        return torch.sigmoid(self.m.mask_head(d2))


def main() -> None:
    model = load_roi_checkpoint(CHECKPOINT, None, device="cpu")
    if model is None:
        raise SystemExit(f"Could not load {CHECKPOINT}")
    config = model.config
    model.eval()
    example = torch.rand(1, 1, 144, 192)
    with torch.no_grad():
        original = torch.sigmoid(model(example)[0])
    export_model = NoConcatRoi(model).eval()
    with torch.no_grad():
        expected = export_model(example)
    print("torch no-concat vs original max diff", float((expected - original).abs().max()))
    torch.onnx.export(
        export_model,
        example,
        OUTPUT,
        input_names=["input"],
        output_names=["mask"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
        dynamic_axes={"input": {2: "height", 3: "width"}, "mask": {2: "mask_height", 3: "mask_width"}},
    )
    onnx.checker.check_model(str(OUTPUT))

    # PyTorch lowers F.pad to a Pad whose ``pads`` input is built with Concat.
    # Replace that vector with a constant initializer and prune the unused
    # shape-construction nodes; otherwise Concat breaks ORT WebGPU.
    graph = onnx.load(str(OUTPUT))
    pad_index = 0
    for node in graph.graph.node:
        if node.op_type != "Pad":
            continue
        pads_name = f"roi_pad_pads_{pad_index}"
        pad_index += 1
        pads = onnx.helper.make_tensor(
            pads_name, onnx.TensorProto.INT64, [8], [0, 0, 0, 0, 0, 2, 0, 0]
        )
        graph.graph.initializer.append(pads)
        node.input[1] = pads_name
    used = {output.name for output in graph.graph.output}
    kept = []
    for node in reversed(graph.graph.node):
        if any(output in used for output in node.output):
            kept.append(node)
            used.update(node.input)
    graph.graph.node.clear()
    graph.graph.node.extend(reversed(kept))
    kept_initializers = [
        initializer for initializer in graph.graph.initializer if initializer.name in used
    ]
    graph.graph.initializer.clear()
    graph.graph.initializer.extend(kept_initializers)
    onnx.checker.check_model(graph)
    onnx.save(graph, str(OUTPUT))
    print("ops", sorted({node.op_type for node in graph.graph.node}))
    print("concat count", sum(node.op_type == "Concat" for node in graph.graph.node))
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
