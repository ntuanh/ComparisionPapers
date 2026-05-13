from ultralytics import YOLO

from config.loader import load_config

from communication.rabbitmq_client import (
    RabbitMQClient
)

from communication.queue_manager import (
    build_tensor_queue_name
)
import pickle

from communication.serializer import (
    deserialize_tensor
)

from runtime.graph_runner import (
    GraphRunner
)

from utils.json_loader import (
    load_json
)


class EdgeNode:

    def __init__(self):

        self.config = load_config()

        server = self.config["server"]

        self.edge_info = load_json(
            "config/edge.json"
        )

        self.client = RabbitMQClient(
            server["host"],
            server["rabbitmq_username"],
            server["rabbitmq_password"]
        )

        self.model = YOLO(
            self.config["model"]["name"]
        ).model.model

    def callback(self,
                 ch,
                 method,
                 properties,
                 body):
        payload = pickle.loads(body)

        partition_idx = \
            payload["partition_idx"]

        message = payload["outputs"]

        outputs = {}

        output_tensor = deserialize_tensor(
            message["output_tensor"]
        )

        outputs[partition_idx] = output_tensor

        for layer_idx, binary_tensor in \
                message[
                    "dependency_tensors"
                ].items():
            outputs[int(layer_idx)] = \
                deserialize_tensor(
                    binary_tensor
                )

        runner = GraphRunner(
            self.model
        )

        final_outputs = \
            runner.run_from_partition(
                outputs,
                partition_idx
            )

        last_output = final_outputs[
            max(final_outputs.keys())
        ]

        print(
            "\nInference Completed"
        )

        print(type(last_output))

    def run(self):
        register_message = {

            "type": "edge",

            **self.edge_info
        }

        self.client.declare_queue(
            "controller_register"
        )

        self.client.publish_json(
            "controller_register",
            register_message
        )

        print(
            "Registered to controller"
        )

        md_id = 1

        es_id = 1

        queue_name = \
            build_tensor_queue_name(
                md_id,
                es_id
            )

        self.client.declare_queue(
            queue_name
        )

        print(
            f"Listening on {queue_name}"
        )

        self.client.consume(
            queue_name,
            self.callback
        )