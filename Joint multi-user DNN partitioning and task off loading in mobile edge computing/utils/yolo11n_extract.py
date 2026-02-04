import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules import Concat, Detect
from dataclasses import dataclass
from typing import List


# --------------------------------------------------
# Dataclasses
# --------------------------------------------------
@dataclass
class LayerProfile:
    name: str
    flops: float
    output_mb: float
    layer_type: str


@dataclass
class DNNProfile:
    name: str
    layers: List[LayerProfile]


# --------------------------------------------------
# Analyzer class
# --------------------------------------------------
class YOLOLayerAnalyzer:
    def __init__(self, model_name="yolo11n.pt", input_size=(1, 3, 640, 640)):
        self.model_name = model_name
        self.model = YOLO(model_name).model
        self.model.eval()
        self.input_size = input_size
        self.layers: List[LayerProfile] = []
        self.hooks = []

    # ---------------- FLOPs formulas ----------------
    def flops_conv(self, layer, output):
        if not isinstance(output, torch.Tensor):
            return 0
        _, c_out, h, w = output.shape
        c_in = layer.in_channels
        k_h, k_w = layer.kernel_size
        return ((c_in * k_h * k_w) +
                (c_in * k_h * k_w - 1) + 1) * c_out * h * w

    def flops_linear(self, layer):
        d_in = layer.in_features
        d_out = layer.out_features
        return (d_in + (d_in - 1) + 1) * d_out

    def flops_batchnorm(self, output):
        if not isinstance(output, torch.Tensor):
            return 0
        _, c, h, w = output.shape
        return 4 * c * h * w

    def flops_silu(self, output):
        if not isinstance(output, torch.Tensor):
            return 0
        _, c, h, w = output.shape
        return 4 * c * h * w

    def output_size_MB(self, output):
        if not isinstance(output, torch.Tensor):
            return 0
        _, c, h, w = output.shape
        return c * h * w * 4 / (1024 * 1024)

    # ---------------- Layer type ----------------
    def get_layer_type(self, module):
        if isinstance(module, nn.Conv2d):
            return "Conv"
        elif isinstance(module, nn.Linear):
            return "Linear"
        elif isinstance(module, nn.BatchNorm2d):
            return "BatchNorm"
        elif isinstance(module, (nn.SiLU, nn.ReLU, nn.LeakyReLU)):
            return "Activation"
        elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
            return "Pool"
        elif isinstance(module, nn.Upsample):
            return "Upsample"
        elif isinstance(module, Concat):
            return "Concat"
        elif isinstance(module, Detect):
            return "Detect"
        else:
            return "Other"

    # ---------------- Hook ----------------
    def hook_fn(self, module, input, output):
        if isinstance(module, nn.Conv2d):
            flops = self.flops_conv(module, output)
        elif isinstance(module, nn.Linear):
            flops = self.flops_linear(module)
        elif isinstance(module, nn.BatchNorm2d):
            flops = self.flops_batchnorm(output)
        elif isinstance(module, nn.SiLU):
            flops = self.flops_silu(output)
        else:
            flops = 0

        output_mb = self.output_size_MB(output) if (
            isinstance(output, torch.Tensor) and output.dim() == 4
        ) else 0

        self.layers.append(
            LayerProfile(
                name=module.__class__.__name__,
                flops=flops,
                output_mb=output_mb,
                layer_type=self.get_layer_type(module)
            )
        )

    # ---------------- Main API ----------------
    def analyze(self) -> DNNProfile:
        # register hooks
        for m in self.model.modules():
            if len(list(m.children())) == 0:
                self.hooks.append(m.register_forward_hook(self.hook_fn))

        # forward pass
        x = torch.randn(*self.input_size)
        with torch.no_grad():
            self.model(x)

        # remove hooks
        for h in self.hooks:
            h.remove()

        return DNNProfile(
            name=self.model_name.replace(".pt", ""),
            layers=self.layers
        )
# if __name__ == "__main__":
#     analyzer = YOLOLayerAnalyzer()
#     profile = analyzer.analyze()
#
#     # example: print first 5 layers
#     for i, layer in enumerate(profile.layers[:5]):
#         print(i, layer)
