class EdgeRunner:

    def __init__(self,
                 model):

        self.model = model

    def run_from_partition(self,
                           partition_idx,
                           current_tensor):

        current = current_tensor

        for idx, layer in enumerate(self.model):

            if idx <= partition_idx:
                continue

            current = layer(current)

        return current