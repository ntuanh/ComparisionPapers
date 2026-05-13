import json
import os


def cache_exists(path):

    return os.path.exists(path)


def load_cache(path):

    with open(path, "r") as f:
        return json.load(f)


def save_cache(path, data):

    with open(path, "w") as f:
        json.dump(data, f, indent=4)