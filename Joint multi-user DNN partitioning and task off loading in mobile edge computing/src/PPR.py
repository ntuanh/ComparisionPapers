from dataclasses import dataclass
from typing import List

from src.Compute import Compute
from src.Hungarian import Hungarian
from collections import Counter
from src.OPP import OPP

# from utils.yolo11n_extract import YOLOLayerAnalyzer
@dataclass
class LayerProfile:
    name: str
    flops: float        # FLOPs
    output_mb: float    # MB
    layer_type: str     # 'Conv', 'Linear', 'Pool', etc.


@dataclass
class DNNProfile:
    name: str
    layers: List[LayerProfile]


class PPR:
    """
    Fixed Partition Point Retain (robust version)
    """

    def __init__(
        self,
        dnn_set,
        bandwidth_set,
        f_local,
        f_edge,
        # compute,
        survive_ratio=1  # >=50% bandwidths
    ):
        self.dnn_set = dnn_set
        self.bandwidth_set = bandwidth_set
        self.f_local = f_local
        self.f_edge = f_edge
        self.compute = Compute()
        self.survive_ratio = survive_ratio

    # ------------------------------------------------
    # Formula (8): execution time after partition
    # ------------------------------------------------
    def execution_time(self, dnn, p, bandwidth):

        # Local execution
        local_flops = sum(l.flops for l in dnn.layers[:p])
        T_local = self.compute.local_time(local_flops, self.f_local)

        # Uplink transmission
        T_upload = self.compute.uplink_time(
            dnn.layers[p].output_mb,
            bandwidth)

        # Edge execution
        edge_flops = sum(l.flops for l in dnn.layers[p:])
        T_edge = self.compute.edge_time(edge_flops, self.f_edge)

        return T_local + T_upload + T_edge

    # ------------------------------------------------
    # FIXED Algorithm 1: PPR
    # ------------------------------------------------
    def run(self):
        RP = set()

        for dnn in self.dnn_set:
            li = len(dnn.layers)
            counter = Counter()

            # -------- iterate bandwidths --------
            for B in self.bandwidth_set:
                T_list = []

                for p in range(li):
                    T_list.append(self.execution_time(dnn, p, B))

                T_avg = sum(T_list) / li

                for p in range(li):
                    layer = dnn.layers[p]
                    if (
                        T_list[p] <= T_avg and
                        layer.layer_type in {"Conv", "Linear", "Pool"}
                    ):
                        counter[p] += 1

            # -------- retain layers surviving most bandwidths --------
            threshold = int(len(self.bandwidth_set) * self.survive_ratio)
            RP_i = {p for p, c in counter.items() if c >= threshold}
            # print(f"\nDNN: {dnn.name}")
            # for p in range(li):
            #     print(f"layer p={p}, counter[p]={counter.get(p, 0)}")
            # print(f'threshold {threshold}')

            # -------- fallback: never return empty --------
            if not RP_i:
                # get p with total time is min
                best_p = min(
                    range(li),
                    key=lambda p: sum(
                        self.execution_time(dnn, p, B)
                        for B in self.bandwidth_set
                    )
                )
                RP_i.add(best_p)

            RP = RP.union(RP_i)

        return RP

