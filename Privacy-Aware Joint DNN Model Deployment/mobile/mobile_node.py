import torch , pickle

import json

from communication.queue_manager import (
    build_assignment_queue_name
)

from ultralytics import YOLO

from config.loader import load_config

from communication.rabbitmq_client import (
    RabbitMQClient
)

from communication.queue_manager import (
    build_tensor_queue_name
)

from mobile.tensor_cache import TensorCache

from mobile.local_runner import LocalRunner

from mobile.partition_sender import (
    build_partition_message
)
from utils.json_loader import (
    load_json
)


class MobileNode:

    def __init__(self):

        self.config = load_config()

        server = self.config["server"]

        self.mobile_info = load_json(
            "config/mobile.json"
        )

        self.client = RabbitMQClient(
            server["host"],
            server["rabbitmq_username"],
            server["rabbitmq_password"]
        )

        self.model = YOLO(
            self.config["model"]["name"]
        ).model.model

        self.tensor_cache = TensorCache()

    def run(self):
        register_message = {

            "type": "mobile",

            **self.mobile_info
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

        while True:
            pass
    def assignment_callback(self,
                            ch,
                            method,
                            properties,
                            body):
        message = json.loads(body)

        partition_idx = \
            message["partition_idx"]

        edge_id = \
            message["edge_id"]

        print(
            f"Received assignment: "
            f"partition={partition_idx}, "
            f"edge={edge_id}"
        )

        self.run_split_inference(
            partition_idx,
            edge_id
        )

    def run_split_inference(self,
                            partition_idx,
                            es_id):
        md_id = 1

        x = torch.randn(
            1,
            3,
            640,
            640
        )

        runner = LocalRunner(
            self.model,
            self.tensor_cache
        )

        output_tensor, dependency_tensors = \
            runner.run_until_partition(
                x,
                partition_idx
            )

        message = build_partition_message(
            output_tensor,
            dependency_tensors
        )

        queue_name = \
            build_tensor_queue_name(
                md_id,
                es_id
            )

        self.client.declare_queue(
            queue_name
        )

        import pickle

        payload = {

            "partition_idx":
                partition_idx,

            "outputs":
                message
        }

        binary = pickle.dumps(payload)

        self.client.publish_binary(
            queue_name,
            binary
        )

        print(
            f"Sent split inference "
            f"to {queue_name}"
        )