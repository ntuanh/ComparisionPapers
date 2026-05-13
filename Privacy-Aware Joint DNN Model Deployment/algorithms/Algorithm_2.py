def estimate_model_utility(server, model):

    return len(server.connected_mds) * 10


def greedy_model_deployment(server,
                            candidate_models):

    remaining = server.storage_capacity

    deployed = []

    scores = []

    for model in candidate_models:

        utility_gain = estimate_model_utility(
            server,
            model
        )

        score = utility_gain / model.total_size

        scores.append((score, model))

    scores.sort(
        reverse=True,
        key=lambda x: x[0]
    )

    for score, model in scores:

        if model.total_size <= remaining:

            deployed.append(model)

            remaining -= model.total_size

    return deployed