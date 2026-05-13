from utils.utility import partition_objective


def find_best_partition(md,
                        server,
                        model,
                        alpha=1.0,
                        xi=1.0):

    best_z = 0

    best_obj = float('inf')

    for z in range(model.num_layers()):

        obj = partition_objective(
            md,
            server,
            model,
            z,
            alpha,
            xi
        )

        if obj < best_obj:

            best_obj = obj
            best_z = z

    return best_z, best_obj