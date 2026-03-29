def select_task(task: int) -> list:
    runs = [
        [ "R03", "R07", "R11"],
        [ "R04", "R08", "R12"],
        [ "R05", "R09", "R13"],
        [ "R06", "R10", "R14"],
    ]
    if task == 0:
        runs = runs
        print("Train model on all tasks")
    elif task in [1, 2, 3, 4]:
        runs = [runs[task - 1]]
        print(f"Train model on task {task}")
    else:
        raise ValueError("Wrong task ID")
    return runs


def select_label(task: int) -> list:
    if task not in [1, 2, 3, 4]:
        raise ValueError("Wrong task number")
    labels = [
        ["Task1 - open and close left or right fist", "left fist", "right fist"],
        ["Task2 - imagine opening and closing left or right fist", "left fist", "right fist"],
        ["Task3 - open and close both fists or both feet", "booth fist", "booth feet"],
        ["Task4 - imagine opening and closing both fists or both feet", "booth fist", "booth feet"]
    ]
    return labels[task - 1]