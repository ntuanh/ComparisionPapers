import torch

from ultralytics import YOLO

from runtime.graph_runner import (
    GraphRunner
)


model = YOLO("yolo11n.pt").model.model

runner = GraphRunner(model)

x = torch.randn(1, 3, 640, 640)

partition = 5

outputs = runner.run_until_partition(
    x,
    partition
)

print(
    f"Finished until partition {partition}"
)

final_outputs = runner.run_from_partition(
    outputs,
    partition
)

print(
    "Full graph execution completed"
)

last_output = final_outputs[
    max(final_outputs.keys())
]

print(type(last_output))

if torch.is_tensor(last_output):

    print(last_output.shape)

elif isinstance(last_output, tuple):

    print("Tuple Output")

    for idx, item in enumerate(last_output):

        if torch.is_tensor(item):

            print(
                f"Tuple[{idx}] Shape: "
                f"{item.shape}"
            )

        elif isinstance(item, list):

            print(
                f"Tuple[{idx}] List Length: "
                f"{len(item)}"
            )