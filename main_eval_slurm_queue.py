"""
TODO: The`slurm_script_path` should be the path to a slurm script adapted to your cluster (defining your env vars etc), which takes the launch command as positional arg, and launches it.
"""

import datetime
import logging
import os
import pprint

from huggingface_hub import HfApi, snapshot_download

from slurm_queue.eval_requests import check_completed_evals, get_eval_requests, is_auto_eval_running, set_eval_request
from slurm_queue.launch_job import launch_job
from slurm_queue.sort_queue import sort_models_by_priority

logging.basicConfig(level=logging.ERROR)
pp = pprint.PrettyPrinter(width=80)

# Where to pull and send requests from
REQUESTS_REPO = "la-leaderboard/requests"

# The tasks you want to launch
TASKS_FILE_2023A = "src/defaults/tasks_2023a.txt"  # TODO: Use our tasks

PENDING_STATUS = "PENDING"
RUNNING_STATUS = "RUNNING"
FINISHED_STATUS = "FINISHED"
FAILED_STATUS = "FAILED"


USER = "lighteval"
SLURM_SCRIPT_8GPU = f"la-leaderboard-backend/slurm_queue/auto_eval_8gpu.slurm"
LOCAL_DIR = f"eval_queue"

NUMBER_OF_JOBS_ON_WEEKDAYS = 40
NUMBER_OF_JOBS_ON_WEEKENDS = 40

# We download the results and requests files
snapshot_download(
    repo_id="la-leaderboard/results", revision="main", local_dir="results", repo_type="dataset", max_workers=60
)
snapshot_download(repo_id=REQUESTS_REPO, revision="main", local_dir=LOCAL_DIR, repo_type="dataset", max_workers=60)


def run_auto_eval():
    """Manages the requests queue and the high level job launching."""
    current_pending_status = [PENDING_STATUS]
    api = HfApi(token=os.getenv("HUGGING_FACE_HUB_TOKEN"))

    # pull the eval dataset from the hub and parse any eval requests
    # check completed evals and set them to finished
    check_completed_evals(
        api=api,
        hf_repo=REQUESTS_REPO,
        checked_status=RUNNING_STATUS,
        completed_status=FINISHED_STATUS,
        failed_status=FAILED_STATUS,
        local_dir=LOCAL_DIR,
    )
    print("====================================")

    # Get all eval request that are PENDING, if you want to run other evals, change this parameter
    eval_requests = get_eval_requests(job_status=current_pending_status, local_dir=LOCAL_DIR)
    # Sort the evals by priority (first submitted first run)
    eval_requests = sort_models_by_priority(models=eval_requests)

    print(f"Found {len(eval_requests)} {','.join(current_pending_status)} eval requests")

    if len(eval_requests) == 0:
        return

    # if there are already evals running, do not queue new ones

    # We schedule jobs differently depending on cluster usage
    today = datetime.datetime.today()
    if today.weekday() in [5, 6]:  # weekends
        max_job_nb = NUMBER_OF_JOBS_ON_WEEKENDS
    else:
        max_job_nb = NUMBER_OF_JOBS_ON_WEEKDAYS

    if is_auto_eval_running(max_job_nb):
        print(f"Already {max_job_nb} evals running, not queuing new ones")
        return

    eval_request = eval_requests[0]
    pp.pprint(eval_request)

    eval_request = launch_job(
        eval_request=eval_request, slurm_script_path=SLURM_SCRIPT_8GPU, tasks_file=TASKS_FILE_2023A, user=USER
    )

    set_eval_request(
        api=api,
        eval_request=eval_request,
        set_to_status=RUNNING_STATUS,
        hf_repo=REQUESTS_REPO,
        local_dir=LOCAL_DIR,
    )


if __name__ == "__main__":
    run_auto_eval()
