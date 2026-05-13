import random

from algorithms.Algorithm_2 import (
    greedy_model_deployment
)

from algorithms.Algorithm_3 import (
    find_best_partition
)

from utils.helpers import (
    random_initialize,
    move_md
)

from utils.utility import (
    evaluate_system_utility
)


def coalition_formation(
        edge_servers,
        mobile_devices,
        models,
        iterations=50):

    random_initialize(
        edge_servers,
        mobile_devices
    )

    current_utility = evaluate_system_utility(
        edge_servers
    )

    for _ in range(iterations):

        md = random.choice(mobile_devices)

        old_server = md.associated_server

        candidate_server = random.choice(
            edge_servers
        )

        if candidate_server == old_server:
            continue

        move_md(
            md,
            old_server,
            candidate_server
        )

        # Algorithm 2
        for es in edge_servers:

            es.deployed_models = \
                greedy_model_deployment(
                    es,
                    models
                )

        # Algorithm 3
        for es in edge_servers:

            for md in es.connected_mds:

                for model in es.deployed_models:
                    z, obj = find_best_partition(
                        md,
                        es,
                        model
                    )

                    # print(
                    #     f"MD {md.id} | "
                    #     f"ES {es.id} | "
                    #     f"Model {model.name} | "
                    #     f"Best Partition = {z} | "
                    #     f"Obj = {obj:.4f}"
                    # )

        new_utility = evaluate_system_utility(
            edge_servers
        )

        if new_utility > current_utility:

            current_utility = new_utility

        else:

            move_md(
                md,
                candidate_server,
                old_server
            )

    return edge_servers