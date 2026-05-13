import torch


class GraphRunner:

    def __init__(self,
                 model):

        self.model = model

    def get_layer_input(self,
                        layer,
                        outputs,
                        previous_output):

        f = layer.f

        # previous layer
        if f == -1:

            return previous_output

        # single dependency
        if isinstance(f, int):

            return outputs[f]

        # multiple dependencies
        if isinstance(f, list):

            tensors = []

            for idx in f:

                if idx == -1:
                    tensors.append(previous_output)

                else:
                    tensors.append(outputs[idx])

            return tensors

        return previous_output

    def run_until_partition(self,
                            x,
                            partition_idx):

        outputs = {}

        previous_output = x

        for idx, layer in enumerate(self.model):

            layer_input = self.get_layer_input(
                layer,
                outputs,
                previous_output
            )

            current_output = layer(layer_input)

            outputs[idx] = current_output

            previous_output = current_output

            if idx == partition_idx:
                break

        return outputs

    def run_from_partition(self,
                           outputs,
                           partition_idx):

        previous_output = outputs[partition_idx]

        for idx, layer in enumerate(self.model):

            if idx <= partition_idx:
                continue

            layer_input = self.get_layer_input(
                layer,
                outputs,
                previous_output
            )

            current_output = layer(layer_input)

            outputs[idx] = current_output

            previous_output = current_output

        return outputs