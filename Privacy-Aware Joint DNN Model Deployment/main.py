import json
import random
import copy


# ---------------------------------------------------------
# 1. LOAD THÔNG SỐ TỪ PROFILING THỰC TẾ (DAG PROFILE)
# ---------------------------------------------------------
def load_real_yolo_profile(profile_path='yolo11n_dag_profile.json'):
    """
    Đọc dữ liệu profiling từ file đã tính toán Skip-Connections.
    File JSON phải có các trường: cut_point, name, flops_giga, payload_size_mb.
    """
    try:
        with open(profile_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file '{profile_path}'. Vui lòng chạy script 'yolo11n_dag_profiler.py' trước!")
        exit(1)


# Nạp dữ liệu
YOLO11N_PROFILE = load_real_yolo_profile()
TOTAL_LAYERS = len(YOLO11N_PROFILE)


def load_data():
    try:
        with open('MD.json', 'r') as f:
            mds = json.load(f)
        with open('ES.json', 'r') as f:
            ess = json.load(f)
        return mds, ess
    except FileNotFoundError as e:
        print(f"LỖI: Thiếu file cấu hình thiết bị - {e}")
        exit(1)


# ---------------------------------------------------------
# 2. ALGORITHM 3: TÌM ĐIỂM CẮT TỐT NHẤT (EXHAUSTIVE SEARCH)
# ---------------------------------------------------------
def find_best_partition(md, es, num_mds_in_es):
    best_delay = float('inf')
    best_split = 0

    # Tài nguyên ES chia đều cho số MD đang kết nối trong cụm
    es_allocated_gflops = es['compute_capacity_gflops'] / num_mds_in_es
    # Băng thông giới hạn bởi uplink của MD hoặc phần chia sẻ của ES
    effective_bw_mbps = min(md['uplink_mbps'], es['bandwidth_mbps'] / num_mds_in_es)
    effective_bw_MBps = effective_bw_mbps / 8.0  # Convert to Megabytes/sec

    for split_index in range(TOTAL_LAYERS):
        profile_data = YOLO11N_PROFILE[split_index]

        # 1. Tính toán GFLOPs chạy trên MD (Local) từ Layer 0 đến điểm cắt hiện tại
        local_flops = sum(l['flops_giga'] for l in YOLO11N_PROFILE[:split_index + 1])
        t_local = local_flops / md['compute_capacity_gflops']

        # 2. Thời gian truyền tải dữ liệu qua mạng (Sử dụng payload_size_mb đã tính DAG)
        tx_data_mb = profile_data['payload_size_mb']
        t_tx = tx_data_mb / effective_bw_MBps if effective_bw_MBps > 0 else float('inf')

        # 3. Tính toán GFLOPs chạy trên ES (Edge) cho các layer còn lại
        edge_flops = sum(l['flops_giga'] for l in YOLO11N_PROFILE[split_index + 1:])
        t_edge = edge_flops / es_allocated_gflops if es_allocated_gflops > 0 else float('inf')

        # Tổng độ trễ = Local + Truyền tải + Edge
        total_delay = t_local + t_tx + t_edge

        if total_delay < best_delay:
            best_delay = total_delay
            best_split = split_index

    return best_split, best_delay


# ---------------------------------------------------------
# TÍNH UTILITY CỦA MỘT LIÊN MINH (COALITION)
# ---------------------------------------------------------
def calculate_coalition_delay(coalition_mds, es):
    num_mds = len(coalition_mds)
    if num_mds == 0: return 0

    total_delay = 0
    for md in coalition_mds:
        _, delay = find_best_partition(md, es, num_mds)
        total_delay += delay
    return total_delay


# ---------------------------------------------------------
# 3. ALGORITHM 1: COALITION FORMATION GAME (PHÂN CỤM)
# ---------------------------------------------------------
def run_coalition_game(mds, ess):
    # Khởi tạo ngẫu nhiên: Gán mỗi MD vào 1 ES bất kỳ
    clusters = {es['es_id']: [] for es in ess}
    for md in mds:
        random_es = random.choice(ess)['es_id']
        clusters[random_es].append(md)

    es_dict = {es['es_id']: es for es in ess}

    converged = False
    iteration = 0

    while not converged:
        converged = True
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")

        # Thử nghiệm Switch Operation cho từng MD
        for md in mds:
            # Tìm ES hiện tại của MD
            current_es_id = next(es_id for es_id, md_list in clusters.items() if md in md_list)
            current_coalition = clusters[current_es_id]

            best_es_id = current_es_id

            # Utility hiện tại của cụm chứa MD
            current_delay = calculate_coalition_delay(current_coalition, es_dict[current_es_id])

            # Thử chuyển MD sang các ES khác để xem tổng delay có giảm không
            for target_es_id in clusters.keys():
                if target_es_id == current_es_id: continue

                target_coalition = clusters[target_es_id]
                target_delay_before = calculate_coalition_delay(target_coalition, es_dict[target_es_id])

                # Giả lập việc di chuyển
                sim_current_coalition = [m for m in current_coalition if m['md_id'] != md['md_id']]
                sim_target_coalition = target_coalition + [md]

                # Tính delay sau khi di chuyển
                current_delay_after = calculate_coalition_delay(sim_current_coalition, es_dict[current_es_id])
                target_delay_after = calculate_coalition_delay(sim_target_coalition, es_dict[target_es_id])

                # Nếu tổng delay của 2 cụm GIẢM đi -> Thực hiện Switch (Lợi ích nhóm)
                if (current_delay_after + target_delay_after) < (current_delay + target_delay_before):
                    best_es_id = target_es_id
                    # Cập nhật mốc so sánh
                    current_delay = current_delay_after
                    target_delay_before = target_delay_after

            if best_es_id != current_es_id:
                # Cập nhật partition (Thực hiện switch)
                clusters[current_es_id] = [m for m in clusters[current_es_id] if m['md_id'] != md['md_id']]
                clusters[best_es_id].append(md)
                converged = False
                print(f"MD {md['md_id']} switched from {current_es_id} to {best_es_id}")
                break  # Break để đánh giá lại từ đầu cấu trúc mới, tránh xung đột logic

    return clusters, es_dict


# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    print(f"Đã nạp thành công Profile YOLO11n với {TOTAL_LAYERS} điểm cắt hợp lệ.")
    mds, ess = load_data()
    print("Bắt đầu mô phỏng Coalition Formation Game...\n")
    final_clusters, es_dict = run_coalition_game(mds, ess)

    print("\n================ FINAL COALITION RESULTS ================")
    for es_id, md_list in final_clusters.items():
        print(f"\n[{es_id}] - {es_dict[es_id]['description']} | Đang phục vụ {len(md_list)} MDs")
        if not md_list:
            print("  -> (Trống)")
            continue

        for md in md_list:
            split_index, delay = find_best_partition(md, es_dict[es_id], len(md_list))
            layer_info = YOLO11N_PROFILE[split_index]

            print(
                f"  -> {md['md_id']:<6} | Cắt tại: {layer_info['name']:<25} | Payload truyền mạng: {layer_info['payload_size_mb']:>5.2f} MB | Tổng Trễ: {delay:.4f}s")