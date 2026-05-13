from communication.queue_manager import (
    build_assignment_queue_name
)


def send_partition_assignment(
        client,
        md_id,
        partition_idx,
        edge_id):

    queue_name = \
        build_assignment_queue_name(
            md_id
        )

    client.declare_queue(
        queue_name
    )

    message = {

        "partition_idx":
            partition_idx,

        "edge_id":
            edge_id
    }

    client.publish_json(
        queue_name,
        message
    )

    print(
        f"Sent assignment to MD {md_id}"
    )