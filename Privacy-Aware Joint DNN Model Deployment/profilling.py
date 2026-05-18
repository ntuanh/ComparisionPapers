import json
import torch
import torch.nn as nn
from ultralytics import YOLO
from thop import profile
import numpy as np

# ---------------------------------------------------------
# 1. ĐỊNH NGHĨA GRAPH DEPENDENCY (DỰA TRÊN YAML)
# Mô tả layer nào cần input từ những layer nào.
# ---------------------------------------------------------
YOLO_DEPENDENCIES = {
    1: [0], 2: [1], 3: [2], 4: [3], 5: [4], 6: [5], 7: [6], 8: [7], 9: [8], 10: [9],
    11: [10],
    12: [11, 6],  # Concat P4 (Cần đầu ra của 11 và 6)
    13: [12],
    14: [13],
    15: [14, 4],  # Concat P3 (Cần đầu ra của 14 và 4)
    16: [15],
    17: [16],
    18: [17, 13],  # Concat Head P4 (Cần 17 và 13)
    19: [18],
    20: [19],
    21: [20, 10],  # Concat Head P5 (Cần 20 và 10)
    22: [21],
    23: [16, 19, 22]  # Detect (Cần 16, 19, 22)
}

YOLO_NAMES = [
    "Conv_P1/2", "Conv_P2/4", "C3k2_B1", "Conv_P3/8", "C3k2_B2",
    "Conv_P4/16", "C3k2_B3", "Conv_P5/32", "C3k2_B4", "SPPF", "C2PSA",
    "nn.Upsample_1", "Concat_P4", "C3k2_H1", "nn.Upsample_2", "Concat_P3",
    "C3k2_H2(P3)", "Conv_H", "Concat_H_P4", "C3k2_H3(P4)",
    "Conv_H_P5", "Concat_H_P5", "C3k2_H4(P5)", "Detect"
]


def calculate_crossing_tensors(cut_point, total_layers=23):
    computed_by_md = set(range(0, cut_point + 1))
    computed_by_es = set(range(cut_point + 1, total_layers + 1))

    crossing_tensors = set()

    for es_layer in computed_by_es:
        if es_layer in YOLO_DEPENDENCIES:
            for required_input in YOLO_DEPENDENCIES[es_layer]:
                if required_input in computed_by_md:
                    crossing_tensors.add(required_input)

    return list(crossing_tensors)


def run_dag_profiler(input_size=(1, 3, 640, 640)):
    model = YOLO('yolo11n.pt')

    # SỬA LỖI Ở ĐÂY: core_model chứa thuật toán routing chuẩn của YOLO
    core_model = model.model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    core_model.to(device)
    core_model.eval()  # Ép về mode suy luận

    dummy_input = torch.randn(*input_size).to(device)
    individual_tensor_sizes = {}

    raw_size_mb = (np.prod(input_size) * 4) / (1024 * 1024)
    individual_tensor_sizes[0] = round(raw_size_mb, 4)

    def get_hook(idx):
        def hook(module, inp, out):
            if isinstance(out, torch.Tensor):
                size = np.prod(out.shape)
            elif isinstance(out, (list, tuple)):
                size = sum([np.prod(o.shape) for o in out if isinstance(o, torch.Tensor)])
            else:
                size = 0
            individual_tensor_sizes[idx] = round((size * 4) / (1024 * 1024), 4)

        return hook

    hooks = []
    # Gắn hook vào từng module con bên trong
    for i, layer in enumerate(core_model.model):
        hooks.append(layer.register_forward_hook(get_hook(i + 1)))

    # SỬA LỖI Ở ĐÂY: Chạy forward trên core_model để giải quyết bài toán Concat
    with torch.no_grad():
        _ = core_model(dummy_input)

    for h in hooks: h.remove()

    # SỬA LỖI Ở ĐÂY: Tính FLOPs cũng phải gọi qua core_model
    total_macs, _ = profile(core_model, inputs=(dummy_input,), verbose=False)
    total_gflops = (total_macs * 2) / 1e9
    total_size = sum(individual_tensor_sizes.values())

    profiling_results = []

    for cut_point in range(0, 23):
        layer_out_size = individual_tensor_sizes.get(cut_point, 0.0)
        flops = total_gflops * (layer_out_size / total_size) if cut_point > 0 else 0.0

        crossing_layers = calculate_crossing_tensors(cut_point)
        payload_size_mb = sum([individual_tensor_sizes.get(l, 0.0) for l in crossing_layers])

        name = "Input_Raw_Image" if cut_point == 0 else f"Layer_{cut_point}_{YOLO_NAMES[cut_point - 1]}"

        profiling_results.append({
            "cut_point": cut_point,
            "name": name,
            "flops_giga": round(flops, 4),
            "payload_size_mb": round(payload_size_mb, 4),
            "crossing_tensors": crossing_layers
        })

    return profiling_results


if __name__ == "__main__":
    print("Khởi động trình phân tích DAG Profiler cho YOLO11n...")
    profile_data = run_dag_profiler()

    output_file = 'yolo11n_dag_profile.json'
    with open(output_file, 'w') as f:
        json.dump(profile_data, f, indent=4)

    print(f"\n================ ĐÃ CẬP NHẬT THEO SKIP-CONNECTIONS ================")
    print("Mẫu 10 điểm cắt đầu tiên:")
    for layer in profile_data[:10]:
        tensors = layer['crossing_tensors']
        print(
            f"Cắt tại {layer['cut_point']:>2} | Payload: {layer['payload_size_mb']:>6.2f} MB | Các tensor phải gửi: {tensors}")