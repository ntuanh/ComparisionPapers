class EdgeServer:
    def __init__(self,
                 server_id,
                 storage_capacity,
                 compute_power):

        self.id = server_id

        # MB
        self.storage_capacity = storage_capacity

        # MFLOPs/s
        self.compute_power = compute_power

        self.deployed_models = []

        self.connected_mds = []