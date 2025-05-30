import json
import subprocess
import sys

from slurm_queue.eval_requests import EvalRequest
from slurm_queue.launch_job import ACCELERATE_CONFIG_FILE, EVAL_SCRIPT_FILE, OUTPUT_PATH, EvalJob

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
        TASKS_HARNESS = ",".join(tasks_todo[model])
        PRECISION = model_precision[model]
        eval_request = EvalRequest(
            model=MODEL,
            precision=PRECISION,
            base_model="",  # TODO: Review arg
            status="",  # TODO: Review arg
            json_filepath="",  # TODO: Review arg
            params=0,
            private=False,
        )

        eval_job_request = EvalJob(
            eval_script_file=EVAL_SCRIPT_FILE,
            accelerate_config_file=ACCELERATE_CONFIG_FILE,
            hub_model=eval_request.model,
            revision=eval_request.revision,
            trust_remote_code=False,
            precision=eval_request.precision,
            model_size_in_b=eval_request.params,
            weight_type=eval_request.weight_type,
            base_model=eval_request.base_model,
            tasks=TASKS_HARNESS,
            output_path=OUTPUT_PATH,
            limit=1,
        )

        command = eval_job_request.build_command()

        try:
            print(f"Executing command: {command}")
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred: {e}")
