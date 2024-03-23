import yaml


def parse_cfg(path):
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)
