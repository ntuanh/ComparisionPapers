def build_tensor_queue_name(
        md_id,
        es_id):

    return (
        f"tensor_md{md_id}_es{es_id}"
    )


def build_result_queue_name(
        md_id):

    return (
        f"result_md{md_id}"
    )

def build_assignment_queue_name(
        md_id):

    return (
        f"assignment_md{md_id}"
    )