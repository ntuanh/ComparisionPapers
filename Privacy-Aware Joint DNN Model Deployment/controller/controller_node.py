import json

from config.loader import load_config

from communication.rabbitmq_client import (
    RabbitMQClient
)


class ControllerNode:

    def __init__(self):

        self.config = load_config()

        server = self.config["server"]

        self.client = RabbitMQClient(
            server["host"],
            server["rabbitmq_username"],
            server["rabbitmq_password"]
        )

        self.alpha = self.config["system"]["alpha"]


        self.registered_mobiles = {}

        self.registered_edges = {}

    def find_best_partition(self,
                            bandwidth,
                            compute_power,
                            edge_power):

        alpha = self.alpha

        best_partition = 0

        best_cost = 999999

        total_layers = 23

        print(
            "\n===== PARTITION SEARCH =====\n"
        )

        for z in range(total_layers):

            # =========================
            # SIMULATED DELAYS
            # =========================

            local_delay = (
                                  (z + 1) * 2
                          ) / compute_power

            upload_delay = (
                                   (total_layers - z) * 5
                           ) / bandwidth

            edge_delay = (
                                 (total_layers - z) * 2
                         ) / edge_power

            # =========================
            # PRIVACY LOSS
            # deeper partition
            # lower privacy leakage
            # =========================

            privacy_loss = (
                    1.0 - (z / total_layers)
            )

            # =========================
            # TOTAL DELAY
            # =========================

            total_delay = (

                    local_delay +

                    upload_delay +

                    edge_delay
            )

            # =========================
            # PAPER OBJECTIVE
            # =========================

            cost = (

                    alpha * total_delay +

                    (1 - alpha) *
                    privacy_loss
            )

            print(
                f"Partition {z} | "
                f"Delay={total_delay:.4f} | "
                f"Privacy={privacy_loss:.4f} | "
                f"Cost={cost:.4f}"
            )

            if cost < best_cost:
                best_cost = cost

                best_partition = z

        print(
            "\n============================"
        )

        print(
            f"BEST PARTITION = "
            f"{best_partition}"
        )

        print(
            "============================\n"
        )

        return best_partition

    def register_callback(self,
                          ch,
                          method,
                          properties,
                          body):

        message = json.loads(body)

        node_type = message["type"]

        # =========================
        # MOBILE REGISTER
        # =========================

        if node_type == "mobile":

            md_id = message["md_id"]

            self.registered_mobiles[
                md_id
            ] = message

            print(
                f"\nMobile {md_id} registered"
            )

        # =========================
        # EDGE REGISTER
        # =========================

        elif node_type == "edge":

            edge_id = message["edge_id"]

            self.registered_edges[
                edge_id
            ] = message

            print(
                f"\nEdge {edge_id} registered"
            )

        # =========================
        # PRINT CURRENT SYSTEM
        # =========================

        print(
            f"Mobiles: "
            f"{list(self.registered_mobiles.keys())}"
        )

        print(
            f"Edges: "
            f"{list(self.registered_edges.keys())}"
        )

        # =========================
        # RUN PAPER OPTIMIZER
        # =========================

        required_mobiles = \
            self.config["system"]["num_mobiles"]

        required_edges = \
            self.config["system"]["num_edges"]

        if (

                len(self.registered_mobiles)
                >= required_mobiles

                and

                len(self.registered_edges)
                >= required_edges
        ):
            print(
                "\nRunning Paper Optimizer..."
            )

            # ==================================
            # CURRENTLY:
            # single MD + single ES
            # ==================================

            md = list(
                self.registered_mobiles.values()
            )[0]

            es = list(
                self.registered_edges.values()
            )[0]

            # ==================================
            # RUN ALGORITHM 3
            # ==================================

            best_partition = \
                self.find_best_partition(

                    bandwidth=
                    md["bandwidth"],

                    compute_power=
                    md["compute_power"],

                    edge_power=
                    es["gpu_power"]
                )

            # ==================================
            # PAPER RESULT
            # ==================================

            print(
                "\n========== PAPER RESULT =========="
            )

            print(
                f"MD {md['md_id']} "
                f"→ ES {es['edge_id']}"
            )

            print(
                f"Best Partition = "
                f"{best_partition}"
            )

            print(
                "==================================\n"
            )

            # ==================================
            # SEND ASSIGNMENT TO MOBILE
            # ==================================

            assignment_queue = (
                f"assignment_md"
                f"{md['md_id']}"
            )

            assignment_message = {

                "partition_idx":
                    best_partition,

                "edge_id":
                    es["edge_id"]
            }

            self.client.declare_queue(
                assignment_queue
            )

            self.client.publish_json(
                assignment_queue,
                assignment_message
            )

            print(
                f"Assignment sent to "
                f"{assignment_queue}"
            )

    def run(self):

        queue_name = "controller_register"

        self.client.declare_queue(
            queue_name
        )

        print(
            "Controller Started"
        )

        print(
            f"Waiting registrations on "
            f"{queue_name}"
        )

        self.client.consume(
            queue_name,
            self.register_callback
        )