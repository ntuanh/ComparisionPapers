import pika
import uuid
import argparse
import yaml
import threading
import torch
import json

import src.Log
from src.RpcClient import RpcClient
from src.Scheduler import Scheduler

QUICK_RUN = True 
SETUP_PTH = "setup.json"

if QUICK_RUN:
    with open(SETUP_PTH, "r") as json_file:
        setup = json.load(json_file)

parser = argparse.ArgumentParser(description="Split learning framework")

parser.add_argument(
    '--layer_id',
    type=int,
    required=True,
    help='ID of layer, start from 1'
)

parser.add_argument(
    '--device',
    type=str,
    required=False,
    help='Device of client'
)

args = parser.parse_args()

# =========================
# LOAD CONFIG
# =========================

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

address = config["rabbit"]["address"]
username = config["rabbit"]["username"]
password = config["rabbit"]["password"]
virtual_host = config["rabbit"]["virtual-host"]

# =========================
# DEVICE
# =========================

if args.device is None:
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using device: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("Using device: CPU")
else:
    device = args.device
    print(f"Using device: {device}")

# =========================
# LOGGER
# =========================

logger = src.Log.Logger(
    "./app.log",
    config['debug-mode']
)

logger.log_info("Application start.")

# =========================
# CUDA STREAMS
# =========================

NUM_THREADS = 3

streams = []

if device == "cuda":
    for i in range(NUM_THREADS):
        streams.append(torch.cuda.Stream(device=device))
else:
    streams = [None] * NUM_THREADS

# =========================
# EXECUTE CLIENT
# =========================

def execute_layer(thread_id, cuda_stream, layer_id):

    # -----------------------------------
    # RabbitMQ connection PER THREAD
    # -----------------------------------
    credentials = pika.PlainCredentials(username, password)
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            address,
            5672,
            f'{virtual_host}',
            credentials
        )
    )
    channel = connection.channel()

    # -----------------------------------
    # UUID
    # -----------------------------------
    if QUICK_RUN:
        client_id = setup[f"md{thread_id}"]["uuid"]
    else:
        client_id = str(uuid.uuid4())

    # Dynamically set queue name based on layer ID
    node_type = "edge" if layer_id == 1 else "cloud"
    queue_name = f"reply_{node_type}_{client_id}"

    print(f"[THREAD {thread_id}] Queue {queue_name}")

    # -----------------------------------
    # REGISTER MESSAGE
    # -----------------------------------
    data = {
        "action": "REGISTER",
        "client_id": client_id,
        "layer_id": layer_id,
        "message": f"Hello from Thread {thread_id}"
    }

    # Only append the partition point if we are on layer 2
    if layer_id == 2:
        data["partition_point"] = setup[f"md{thread_id}"]["partition_point"]

    src.Log.print_with_color(
        f"[>>>] Thread {thread_id} sending registration...",
        "red"
    )

    # -----------------------------------
    # EXECUTION CORE
    # -----------------------------------
    def run_client():
        scheduler = Scheduler(
            client_id,
            layer_id,
            channel,
            device
        )

        logger.log_debug(
            f"thread_id={thread_id}, "
            f"client_id={client_id}, "
            f"layer={layer_id}, "
            f"device={device}"
        )

        client = RpcClient(
            client_id,
            layer_id,
            channel,
            logger,
            scheduler.inference_func,
            device
        )

        client.send_to_server(data)
        client.wait_response()

    # -----------------------------------
    # CUDA STREAM
    # -----------------------------------
    if device == "cuda":
        with torch.cuda.stream(cuda_stream):
            run_client()
        torch.cuda.synchronize()
    else:
        run_client()

    # -----------------------------------
    # CLOSE CONNECTION
    # -----------------------------------
    connection.close()
    print(f"[THREAD {thread_id}] Finished")

# =========================
# MAIN
# =========================

if __name__ == "__main__":
    threads = []

    for i in range(NUM_THREADS):
        t = threading.Thread(
            target=execute_layer,
            args=(i, streams[i], args.layer_id)
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print("All threads completed.")