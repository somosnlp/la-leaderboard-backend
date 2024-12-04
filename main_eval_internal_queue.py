import json
import sys

from basic_queue.backend.manage_requests import EvalRequest
from basic_queue.backend.run_eval_suite_harness import run_evaluation
from basic_queue.envs import (
    BATCH_SIZE,
    DEVICE,
    EVAL_RESULTS_PATH_BACKEND,
    LEADERBOARD_GROUP,
    LIMIT,
    LOGS_REPO,
    NUM_FEWSHOT,
    RESULTS_REPO,
)

if __name__ == "__main__":

    if len(sys.argv) > 1:
        tasks_todo_path = sys.argv[1]
    else:
        tasks_todo_path = "internal_queue/tasks_todo.json"

    with open(tasks_todo_path, "r") as f:
        tasks_todo = json.load(f)
    with open("internal_queue/model_precision.json", "r") as f:
        model_precision = json.load(f)

    for model in tasks_todo:
        MODEL = model
        TASKS_HARNESS = tasks_todo[model]
        PRECISION = model_precision[model]
        EVAL_REQUEST = EvalRequest(
            model=MODEL,
            precision=PRECISION,
            base_model="",  # TODO: Review arg
            status="",  # TODO: Review arg
            json_filepath="",  # TODO: Review arg
        )

        run_evaluation(
            eval_request=EVAL_REQUEST,
            task_names=TASKS_HARNESS,
            leaderboard_group=LEADERBOARD_GROUP,
            num_fewshot=NUM_FEWSHOT,
            batch_size=BATCH_SIZE,
            device=DEVICE,
            local_dir=EVAL_RESULTS_PATH_BACKEND,
            results_repo=RESULTS_REPO,
            logs_repo=LOGS_REPO,
            limit=LIMIT,
        )
