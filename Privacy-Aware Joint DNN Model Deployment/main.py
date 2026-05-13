import sys

from mobile.mobile_node import MobileNode
from edge.edge_node import EdgeNode
from controller.controller_node import ControllerNode


if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "Usage: python main.py "
            "[mobile|edge|controller]"
        )

        exit()

    mode = sys.argv[1]

    if mode == "mobile":

        MobileNode().run()

    elif mode == "edge":

        EdgeNode().run()

    elif mode == "controller":

        ControllerNode().run()

    else:

        print("Invalid mode")