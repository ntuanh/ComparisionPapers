import time
import json

import torch
from ultralytics import YOLO

from entities.dnn_model import (
    Layer,
    DNNModel
)


class YOLO11Profiler:

    def __init__(self,
                 model_path="yolo11n.pt",
                 image_size=640,
                 device=None):

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        print(f"Using device: {self.device}")

        self.yolo = YOLO(model_path)

        self.model = self.yolo.model.model.to(device)

        self.image_size = image_size

        self.profile_results = []

    def fake_privacy_score(self,
                           layer_idx,
                           total_layers):

        # deeper layer -> lower privacy leakage
        score = 1.0 - (layer_idx / total_layers)

        return max(score, 0.05)

    def profile(self):

        x = torch.randn(
            1,
            3,
            self.image_size,
            self.image_size
        ).to(self.device)

        self.profile_results = []

        print(f"Total Layers: {len(self.model)}")

        # forward hooks
        hooks = []

        layer_outputs = {}

        def make_hook(idx, layer):

            def hook(module,
                     input,
                     output):
                if not torch.is_tensor(output):
                    return

                # tensor size
                tensor_size_bytes = (
                        output.numel() *
                        output.element_size()
                )

                tensor_size_kb = (
                        tensor_size_bytes / 1024
                )

                layer_outputs[idx] = {

                    "layer_index": idx,

                    "layer_name":
                        layer.__class__.__name__,

                    "tensor_shape":
                        list(output.shape),

                    "tensor_size_kb":
                        tensor_size_kb
                }

            return hook

        # register hooks
        for idx, layer in enumerate(self.model):
            hooks.append(
                layer.register_forward_hook(
                    make_hook(idx, layer)
                )
            )

        # warmup
        with torch.no_grad():

            for _ in range(3):
                _ = self.yolo.model(x)

        if self.device == "cuda":
            torch.cuda.synchronize()

        # runtime profiling
        runtimes = {}

        with torch.no_grad():

            current_input = x

            for idx, layer in enumerate(self.model):

                # use fresh input flow
                start = time.perf_counter()

                _ = self.yolo.model(x)

                if self.device == "cuda":
                    torch.cuda.synchronize()

                end = time.perf_counter()

                total_runtime_ms = (
                        (end - start) * 1000
                )

                runtimes[idx] = (
                        total_runtime_ms /
                        len(self.model)
                )

        # remove hooks
        for h in hooks:
            h.remove()

        total_layers = len(layer_outputs)

        # print("\n========== PROFILE RESULTS ==========")

        for idx in sorted(layer_outputs.keys()):
            result = layer_outputs[idx]

            privacy_loss = self.fake_privacy_score(
                idx,
                total_layers
            )

            layer_result = {

                "layer_index":
                    idx,

                "layer_name":
                    result["layer_name"],

                "runtime_ms":
                    runtimes[idx],

                "tensor_shape":
                    result["tensor_shape"],

                "tensor_size_kb":
                    result["tensor_size_kb"],

                "privacy_loss":
                    privacy_loss
            }

            self.profile_results.append(
                layer_result
            )

        #     print("\n---------------------------")
        #
        #     print(
        #         f"Layer {idx}"
        #     )
        #
        #     print(
        #         f"Name: "
        #         f"{layer_result['layer_name']}"
        #     )
        #
        #     print(
        #         f"Runtime: "
        #         f"{layer_result['runtime_ms']:.4f} ms"
        #     )
        #
        #     print(
        #         f"Tensor Shape: "
        #         f"{layer_result['tensor_shape']}"
        #     )
        #
        #     print(
        #         f"Tensor Size: "
        #         f"{layer_result['tensor_size_kb']:.2f} KB"
        #     )
        #
        #     print(
        #         f"Privacy Loss: "
        #         f"{layer_result['privacy_loss']:.4f}"
        #     )
        #
        # print("\n====================================")

        return self.profile_results

    def export_json(self,
                    save_path="profiling/profile_output.json"):

        with open(save_path, "w") as f:

            json.dump(
                self.profile_results,
                f,
                indent=4
            )

        print(f"\nSaved profiling to {save_path}")

    def build_dnn_model(self):

        layers = []

        total_model_size = 0

        for result in self.profile_results:

            layer = Layer(
                name=result["layer_name"],

                compute_load=result["runtime_ms"],

                comm_load=result["tensor_size_kb"],

                privacy_loss=result["privacy_loss"]
            )

            layers.append(layer)

            total_model_size += result["tensor_size_kb"]

        dnn_model = DNNModel(
            name="YOLO11n",

            total_size=total_model_size / 1024,

            layers=layers
        )

        return dnn_model


# if __name__ == "__main__":
#
#     profiler = YOLO11Profiler()
#
#     profiler.profile()
#
#     profiler.export_json()
#
#     yolo_model = profiler.build_dnn_model()
#
#     print("\nYOLO11n DNN Model Created")
#
#     print(
#         f"Layers: "
#         f"{yolo_model.num_layers()}"
#     )
#
#     print(
#         f"Total Size: "
#         f"{yolo_model.total_size:.2f} MB"
#     )