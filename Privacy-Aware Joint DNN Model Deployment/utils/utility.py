from utils.delay import (
    compute_local_delay,
    compute_upload_delay,
    compute_edge_delay
)


def partition_objective(md,
                        server,
                        model,
                        z,
                        alpha=1.0,
                        xi=1.0):

    local_delay = compute_local_delay(
        md,
        model,
        z
    )

    upload_delay = compute_upload_delay(
        md,
        model,
        z
    )

    edge_delay = compute_edge_delay(
        server,
        model,
        z
    )

    total_delay = max(
        local_delay,
        max(upload_delay, edge_delay)
    )

    privacy = (
        model.layers[z].privacy_loss
    )

    obj = (
        alpha * total_delay +
        xi * privacy
    )

    return obj


def evaluate_system_utility(
        edge_servers):

    utility = 0

    for es in edge_servers:

        utility -= len(
            es.connected_mds
        )

    return utility