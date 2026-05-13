class Layer:
    def __init__(self,
                 name,
                 compute_load,
                 comm_load,
                 privacy_loss):

        self.name = name

        # MFLOPs
        self.compute_load = compute_load

        # KB
        self.comm_load = comm_load

        # privacy leakage possibility
        self.privacy_loss = privacy_loss


class DNNModel:
    def __init__(self,
                 name,
                 total_size,
                 layers):

        self.name = name

        # MB
        self.total_size = total_size

        self.layers = layers

    def num_layers(self):
        return len(self.layers)