import random


def move_md(md,
            old_server,
            new_server):

    # remove safely
    if md in old_server.connected_mds:
        old_server.connected_mds.remove(md)

    # avoid duplicate append
    if md not in new_server.connected_mds:
        new_server.connected_mds.append(md)

    md.associated_server = new_server


def random_initialize(edge_servers,
                      mobile_devices):

    # clear old states first
    for es in edge_servers:
        es.connected_mds.clear()

    for md in mobile_devices:

        es = random.choice(edge_servers)

        es.connected_mds.append(md)

        md.associated_server = es