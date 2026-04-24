import json


VALID_CLASSIFIERS = [
    "CSP_Logreg", "CSP_MyLogreg", "MyCSP_MyLogreg",
    "CSP_LDA", "MyCSP_LDA", "CSP_SVM", "CSP_XGBOOST", "CSP_MLP"
]
RUNS_PER_TASK = {"1": 3, "2": 3, "3": 3, "4": 3, "5": 6, "6": 6}


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if "task" not in config:
        raise ValueError("config error: missing 'task'")

    for k, v in config["task"].items():
        if k not in RUNS_PER_TASK:
            raise ValueError(f"config error: invalid task index '{k}', must be 1-6")

        for field in ["n_comp", "classifier", "search_param", "model", "runs", "title", "labels"]:
            if field not in v:
                raise ValueError(f"config error: task {k} missing '{field}'")

        if not isinstance(v["n_comp"], int) or v["n_comp"] <= 0:
            raise ValueError(f"config error: task {k} 'n_comp' must be a positive integer")

        if v["classifier"] not in VALID_CLASSIFIERS:
            raise ValueError(f"config error: task {k} invalid classifier '{v['classifier']}', "
                             f"must be one of {VALID_CLASSIFIERS}")

        if not isinstance(v["search_param"], bool):
            raise ValueError(f"config error: task {k} 'search_param' must be a boolean")

        if not isinstance(v["runs"], list) or len(v["runs"]) != RUNS_PER_TASK[k]:
            raise ValueError(f"config error: task {k} 'runs' must be a list of "
                             f"{RUNS_PER_TASK[k]} elements")

        if not isinstance(v["labels"], list) or len(v["labels"]) != 2:
            raise ValueError(f"config error: task {k} 'labels' must be a list of 2 strings")

    return config
