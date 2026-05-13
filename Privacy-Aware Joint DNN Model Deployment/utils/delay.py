def compute_local_delay(md, model, z):

    total_compute = 0

    for i in range(z + 1):
        total_compute += model.layers[i].compute_load

    return total_compute / md.local_compute_power


def compute_upload_delay(md, model, z):

    comm_load = model.layers[z].comm_load

    return comm_load / md.uplink_rate


def compute_edge_delay(server, model, z):

    total_compute = 0

    for i in range(z + 1, model.num_layers()):
        total_compute += model.layers[i].compute_load

    return total_compute / server.compute_power