import torch
from ultralytics import YOLO
from collections import OrderedDict

from utils.Utils import Compute

# =========================
# Your formulas
# =========================

# class Compute:
#     def Fl(self, din, dout):
#         return 2 * din * dout
#
#     def Fc(self, cin, cout, kw, kh, w, h):
#         return 2 * cin * kw * kh * cout * w * h
#
#     def Dl(self, cin, w, h):
#         return cin * w * h * 4 / (1024 * 1024)


compute = Compute()

# =========================
# Storage
# =========================

layer_flops = OrderedDict()
layer_output_size = OrderedDict()

# =========================
# Hook function
# =========================

def hook_fn(name, module):
    def hook(module, inputs, output):

        # Handle tuple outputs (YOLO Detect, etc.)
        if isinstance(output, (list, tuple)):
            output = output[0]

        if not isinstance(output, torch.Tensor):
            return

        b, c, h, w = output.shape

        flops = 0

        # -------- Conv2d --------
        if isinstance(module, torch.nn.Conv2d):
            cin = module.in_channels
            cout = module.out_channels
            kh, kw = module.kernel_size
            groups = module.groups
            cin = cin // groups
            flops = compute.Fc(cin, cout, kw, kh, w, h)

        # -------- Linear --------
        elif isinstance(module, torch.nn.Linear):
            flops = compute.Fl(module.in_features, module.out_features)

        # -------- Pool / Upsample / Concat --------
        elif isinstance(module, (
            torch.nn.MaxPool2d,
            torch.nn.AdaptiveAvgPool2d,
            torch.nn.Upsample
        )):
            flops = c * h * w  # lightweight ops

        # -------- Activation --------
        elif isinstance(module, (
            torch.nn.SiLU,
            torch.nn.ReLU
        )):
            flops = c * h * w

        # -------- Default --------
        else:
            flops = 0

        layer_flops[name] = flops
        layer_output_size[name] = compute.Dl(c, w, h)

    return hook


# =========================
# Load YOLO11n model
# =========================

model = YOLO("yolo11n.yaml").model
model.eval()

# Register hooks
for name, m in model.named_modules():
    if isinstance(m, (
        torch.nn.Conv2d,
        torch.nn.Linear,
        torch.nn.MaxPool2d,
        torch.nn.AdaptiveAvgPool2d,
        torch.nn.Upsample,
        torch.nn.SiLU,
        torch.nn.ReLU
    )):
        m.register_forward_hook(hook_fn(name, m))

# =========================
# Run dummy inference
# =========================

x = torch.randn(1, 3, 640, 640)
with torch.no_grad():
    model(x)

# =========================
# RETURN TWO LISTS
# =========================

output_sizes_mb = list(layer_output_size.values())
flops_per_layer = list(layer_flops.values())

print(f"Layers counted: {len(output_sizes_mb)}")
print("Output sizes (MB):", output_sizes_mb[:5], "...")
print("FLOPs:", flops_per_layer[:5], "...")
