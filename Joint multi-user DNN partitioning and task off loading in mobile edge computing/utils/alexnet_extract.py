# alexnet_extract_21layers.py

import torch
import torchvision.models as models
import yaml
import matplotlib.pyplot as plt
from collections import OrderedDict

# =========================
# FLOPs & Data Size Models
# =========================

def conv_flops(layer, output):
    c_out, h, w = output.shape[1:]
    k_h, k_w = layer.kernel_size
    c_in = layer.in_channels
    groups = layer.groups
    cin_per_group = c_in // groups

    flops_per_element = (
        cin_per_group * k_h * k_w +          # multiplications
        (cin_per_group * k_h * k_w - 1) +    # additions
        1                                   # bias
    )
    return flops_per_element * c_out * h * w


def linear_flops(layer):
    in_f = layer.in_features
    out_f = layer.out_features
    return (in_f + (in_f - 1) + 1) * out_f


def relu_flops(output):
    return output.numel()   # 1 comparison per element


def pool_flops(output):
    return output.numel()   # comparisons


def adaptive_avgpool_flops(output):
    return output.numel()   # averaging ops (simplified)


def dropout_flops(output):
    return 0                # counted as a layer, no FLOPs


def output_size_mb(output):
    # Eq. (3), FP32
    return output.numel() * 4 / (1024 * 1024)


# =========================
# Hook logic
# =========================

layer_stats = OrderedDict()

def hook_fn(name):
    def hook(layer, input, output):

        if isinstance(layer, torch.nn.Conv2d):
            flops = conv_flops(layer, output)

        elif isinstance(layer, torch.nn.Linear):
            flops = linear_flops(layer)

        elif isinstance(layer, torch.nn.ReLU):
            flops = relu_flops(output)

        elif isinstance(layer, torch.nn.MaxPool2d):
            flops = pool_flops(output)

        elif isinstance(layer, torch.nn.AdaptiveAvgPool2d):
            flops = adaptive_avgpool_flops(output)

        elif isinstance(layer, torch.nn.Dropout):
            flops = dropout_flops(output)

        else:
            return

        layer_stats[name] = {
            "flops_g": flops / 1e9,
            "output_mb": output_size_mb(output)
        }

    return hook


# =========================
# Main
# =========================

def main():

    model = models.alexnet(pretrained=False)
    model.eval()

    # Register hooks (paper-style layer counting)
    for name, layer in model.named_modules():
        if isinstance(layer, (
            torch.nn.Conv2d,
            torch.nn.ReLU,
            torch.nn.MaxPool2d,
            torch.nn.AdaptiveAvgPool2d,
            torch.nn.Dropout,
            torch.nn.Linear
        )):
            layer.register_forward_hook(hook_fn(name))

    # Forward pass
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        model(x)

    # =========================
    # Extract lists (bar chart)
    # =========================
    names = list(layer_stats.keys())
    output_mb = [v["output_mb"] for v in layer_stats.values()]
    flops_g = [v["flops_g"] for v in layer_stats.values()]

    print(f"Extracted layers: {len(names)}")
    for n in names:
        print(n)

    # =========================
    # Save YAML (paper-correct)
    # =========================
    yaml_data = {
        "model": "AlexNet",
        "input": [3, 224, 224],
        "num_layers": len(names),
        "layers": []
    }

    for name, v in layer_stats.items():
        yaml_data["layers"].append({
            "name": name,
            "flops_g": round(v["flops_g"], 6),
            "output_mb": round(v["output_mb"], 6)
        })

    with open("alexnet_21layers.yaml", "w") as f:
        yaml.dump(yaml_data, f, sort_keys=False)

    print("Saved: alexnet_21layers.yaml")

    # =========================
    # Plot bar chart (Fig. 3 style)
    # =========================
    fig, ax1 = plt.subplots(figsize=(15, 6))

    ax1.bar(names, output_mb, color="red", alpha=0.7)
    ax1.set_ylabel("Size of output data (MB)", color="red")
    ax1.tick_params(axis="y", labelcolor="red")
    ax1.set_xticklabels(names, rotation=45, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(names, flops_g, color="blue", marker="o")
    ax2.set_ylabel("FLOPs (GFLOPs)", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")

    plt.title("AlexNet: Calculation amount and output data size of each layer")
    plt.tight_layout()
    plt.savefig("alexnet_21layers_bar.png", dpi=300)
    plt.show()

    print("Saved: alexnet_21layers_bar.png")


if __name__ == "__main__":
    main()
