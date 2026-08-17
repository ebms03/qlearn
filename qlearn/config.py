import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def kwargs(kwargs, additional_removes=()):
    return {
        k: v for k, v in kwargs.items() if k != "type" and k not in additional_removes
    }
