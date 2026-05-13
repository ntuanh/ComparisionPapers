from ultralytics import YOLO


class DependencyGraph:

    def __init__(self,
                 model_path="yolo11n.pt"):

        self.yolo = YOLO(model_path)

        self.model = self.yolo.model.model

        self.graph = self.build_graph()

    def build_graph(self):

        graph = {}

        for idx, layer in enumerate(self.model):

            dependencies = layer.f

            # normalize single int
            if isinstance(dependencies, int):
                dependencies = [dependencies]

            clean_dependencies = []

            for dep in dependencies:

                # ignore previous-layer shortcut
                if dep == -1:
                    continue

                clean_dependencies.append(dep)

            graph[idx] = clean_dependencies

        return graph

    def get_required_tensors(self,
                             partition_idx):

        required = set()

        # check future layers
        for future_layer, deps in self.graph.items():

            # only layers after partition
            if future_layer <= partition_idx:
                continue

            for dep in deps:

                # dependency exists on mobile side
                if dep <= partition_idx:

                    required.add(dep)

        return sorted(list(required))

    def print_graph(self):

        print(
            "\n====== YOLO Dependency Graph ======\n"
        )

        for layer_idx, deps in self.graph.items():

            print(
                f"Layer {layer_idx} "
                f"depends on {deps}"
            )


if __name__ == "__main__":

    graph = DependencyGraph()

    graph.print_graph()

    partition = 5

    required = graph.get_required_tensors(
        partition
    )

    print(
        f"\nPartition {partition} "
        f"must send tensors: {required}"
    )