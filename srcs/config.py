import json


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config