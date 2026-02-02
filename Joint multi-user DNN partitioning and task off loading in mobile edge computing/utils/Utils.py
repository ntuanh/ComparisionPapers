import yaml

def read_yaml(file_name):
    with open(file_name, "r") as f:
        cfg = yaml.safe_load(f)
        return cfg