def clear_queue(channel,
                queue_name):

    channel.queue_purge(
        queue=queue_name
    )