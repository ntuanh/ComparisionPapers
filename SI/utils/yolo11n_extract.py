import torch
import torch.nn as nn
from ultralytics import YOLO


# --------------------------------------------------
# Analyzer class
# --------------------------------------------------
class YOLOLayerAnalyzer:
    def __init__(self, model_name="yolo11n.pt", input_size=(1, 3, 640, 640)):
        self.model_name = model_name
        self.model = YOLO(model_name).model
        self.model.eval()
        self.input_size = input_size

        # results
        self.flops_layers = []
        self.output_size_layers = []

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

    # ---------------- Hook ----------------
    def hook_fn(self, module, input, output):
        # ---- FLOPs ----
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

        # ---- Output size ----
        if isinstance(output, torch.Tensor) and output.dim() == 4:
            output_mb = self.output_size_MB(output)
        else:
            output_mb = 0

        self.flops_layers.append(flops)
        self.output_size_layers.append(output_mb)

    # ---------------- Main API ----------------
    def analyze(self):
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

        return self.flops_layers, self.output_size_layers[:-1]

# analyzer = YOLOLayerAnalyzer("yolo11n.pt")
# flops_layers, output_size_layers = analyzer.analyze()
#
# print(len(flops_layers), len(output_size_layers))
# print(flops_layers[:5])
# print(output_size_layers[:5])
# print(len(flops_layers))
# print(len(output_size_layers))
