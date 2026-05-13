import io
import torch


def serialize_tensor(tensor):

    buffer = io.BytesIO()

    torch.save(tensor, buffer)

    return buffer.getvalue()


def deserialize_tensor(binary):

    buffer = io.BytesIO(binary)

    return torch.load(buffer)