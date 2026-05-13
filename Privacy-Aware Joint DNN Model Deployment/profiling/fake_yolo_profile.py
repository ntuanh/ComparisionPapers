from entities.dnn_model import (
    Layer,
    DNNModel
)


def create_fake_yolo11n():

    layers = [

        Layer(
            name="Conv1",
            compute_load=20,
            comm_load=12000,
            privacy_loss=1.0
        ),

        Layer(
            name="Conv2",
            compute_load=40,
            comm_load=8000,
            privacy_loss=0.9
        ),

        Layer(
            name="C3_Block_1",
            compute_load=80,
            comm_load=4000,
            privacy_loss=0.7
        ),

        Layer(
            name="C3_Block_2",
            compute_load=120,
            comm_load=2000,
            privacy_loss=0.5
        ),

        Layer(
            name="SPPF",
            compute_load=150,
            comm_load=800,
            privacy_loss=0.3
        ),

        Layer(
            name="Detect",
            compute_load=50,
            comm_load=100,
            privacy_loss=0.1
        ),
    ]

    model = DNNModel(
        name="YOLO11n",
        total_size=7,
        layers=layers
    )

    return model