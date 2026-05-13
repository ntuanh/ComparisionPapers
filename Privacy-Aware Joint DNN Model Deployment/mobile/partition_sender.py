from communication.serializer import (
    serialize_tensor
)


def build_partition_message(
        output_tensor,
        dependency_tensors):

    serialized_dependencies = {}

    for k, v in dependency_tensors.items():

        serialized_dependencies[k] = \
            serialize_tensor(v)

    message = {

        "output_tensor":
            serialize_tensor(output_tensor),

        "dependency_tensors":
            serialized_dependencies
    }

    return message