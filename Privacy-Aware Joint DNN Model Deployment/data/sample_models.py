from entities.dnn_model import (
    Layer,
    DNNModel
)


def create_sample_models():

    vgg_layers = [

        Layer("Conv1", 100, 12000, 1.0),
        Layer("Conv2", 200, 6000, 0.8),
        Layer("Conv3", 300, 3000, 0.4),
        Layer("FC1", 50, 100, 0.1),
    ]

    vgg = DNNModel(
        "VGG19",
        500,
        vgg_layers
    )

    return [vgg]