class TensorCache:

    def __init__(self):

        self.cache = {}

    def store(self,
              layer_idx,
              tensor):

        self.cache[layer_idx] = tensor

    def get(self,
            layer_idx):

        return self.cache.get(layer_idx)

    def clear(self):

        self.cache.clear()