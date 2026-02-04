
class OPP :
    """

    :return: m : ( Cost minimum , best partition point correspond)
    denote m for MD index
    CM = {
    0: {
        0: (C_00, p_00),  # MD 0 -> ES 0
        1: (C_01, p_01)   # MD 0 -> ES 1
    },
    1: {
        0: (C_10, p_10),  # MD 1 -> ES 0
        1: (C_11, p_11)   # MD 1-> ES 1
    }
    }


    """
    def __init__(
            self ,
            dnn_set,
            bandwidth_set,
            f_local_set,
            f_edge_set,
            RP_set,
            compute):
        self.dnn_set = dnn_set
        self.bandwidth_set = bandwidth_set
        self.f_local_set = f_local_set
        self.f_edge_set = f_edge_set
        self.RP_set = RP_set
        self.compute = compute

    def execution_time(self, dnn, p, f_local, f_edge):
        """
        T_i^k for given partition p and edge server k
        """

        # Local execution
        local_flops = sum(l.flops for l in dnn.layers[:p])
        T_local = self.compute.local_time(local_flops, f_local)

        # Upload
        T_upload = self.compute.uplink_time(
            dnn.layers[p].output_mb,
            self.bandwidth_set)

        # Edge execution
        edge_flops = sum(l.flops for l in dnn.layers[p:])
        T_edge = self.compute.edge_time(edge_flops, f_edge)

        return T_local + T_upload + T_edge


    def run(self):
        """
        Compute cost matrix CM
        :return: CM (dict)
        """
        CM = {}

        for i, dnn in enumerate(self.dnn_set):
            CM[i] = {}

            for k, f_edge in self.f_edge_set.items():

                best_cost = float("inf")
                best_p = None

                # Line 4–12: iterate candidate partition points
                for p in self.RP_set[i]:

                    T = self.execution_time(
                        dnn,
                        p,
                        self.f_local_set[i],
                        f_edge
                    )

                    C = T  # since alpha=1, E=0

                    if C < best_cost:
                        best_cost = C
                        best_p = p

                # Line 9–10: record cost matrix
                CM[i][k] = (best_cost, best_p)

        return CM


