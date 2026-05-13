from profiling.dependency_graph import (
    DependencyGraph
)


class LocalRunner:

    def __init__(self,
                 model,
                 tensor_cache):

        self.model = model

        self.tensor_cache = tensor_cache

        self.dependency_graph = \
            DependencyGraph()

    def run_until_partition(self,
                            x,
                            partition_idx):

        outputs = {}

        previous_output = x

        for idx, layer in enumerate(self.model):

            f = layer.f

            # previous layer
            if f == -1:

                layer_input = previous_output

            # single dependency
            elif isinstance(f, int):

                layer_input = outputs[f]

            # multi dependency
            else:

                layer_input = []

                for dep in f:

                    if dep == -1:
                        layer_input.append(
                            previous_output
                        )

                    else:
                        layer_input.append(
                            outputs[dep]
                        )

            current_output = layer(
                layer_input
            )

            outputs[idx] = current_output

            self.tensor_cache.store(
                idx,
                current_output
            )

            previous_output = current_output

            if idx == partition_idx:
                break

        required_layers = \
            self.dependency_graph \
                .get_required_tensors(
                    partition_idx
                )

        dependency_tensors = {}

        for layer_idx in required_layers:

            dependency_tensors[layer_idx] = \
                self.tensor_cache.get(
                    layer_idx
                )

        return current_output, dependency_tensors