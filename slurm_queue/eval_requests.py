"""
Parse eval request and change their status depending on whether things have run.
"""

import glob
import json
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional

from huggingface_hub import HfApi

TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")


@dataclass
class EvalRequest:
    model: str
    private: bool
    status: str
    json_filepath: str
    weight_type: str = "Original"
    model_type: str = ""  # pretrained, finetuned, with RL
    precision: str = ""  # float16, bfloat16, 8bit, 4bit, GPTQ
    base_model: Optional[str] = None
    revision: str = "main"
    submitted_time: Optional[str] = (
        "2022-05-18T11:40:22.519222"  # random date just so that we can still order requests by date
    )
    size: Optional[int] = None
    job_id: Optional[str] = None
    model_type: Optional[str] = None
    likes: Optional[int] = None
    license: Optional[str] = None
    params: Optional[int] = None
    job_start_time: Optional[str] = None


def set_eval_request(
    api: HfApi, eval_request: EvalRequest, set_to_status: str, hf_repo: str, local_dir: str = "./evals/"
):
    """Updates a given eval request with its new status on the hub (running, completed, failed, ...)"""

    json_filepath = eval_request.json_filepath

    with open(json_filepath) as fp:
        data = json.load(fp)

    data["status"] = set_to_status
    data["job_id"] = eval_request.job_id
    data["job_start_time"] = eval_request.job_start_time

    with open(json_filepath, "w") as f:
        f.write(json.dumps(data, indent=2))

    api.upload_file(
        path_or_fileobj=json_filepath,
        path_in_repo=json_filepath.replace(local_dir, ""),
        repo_id=hf_repo,
        repo_type="dataset",
    )


def get_eval_requests(job_status: list, local_dir: str = "./evals/") -> List[EvalRequest]:
    """Get all pending evaluation requests and return a list in which private
    models appearing first, followed by public models sorted by the number of
    likes.

    Args:
        json_files (`List[str]`): a list of evaluation request JSON files
        job_status (`list`): a string indicating job status ("PENDING",
        "RUNNING", "FINISHED")

    Returns:
        `List[EvalRequest]`: a list of model info dicts.
    """

    json_files = glob.glob(f"{local_dir}/**/*.json", recursive=True)

    eval_requests = []
    for json_filepath in json_files:
        with open(json_filepath) as fp:
            data = json.load(fp)
        if data["status"] in job_status:
            data["json_filepath"] = json_filepath
            eval_request = EvalRequest(**data)
            eval_requests.append(eval_request)

    return eval_requests


def list_slurm_jobs() -> List[Dict[str, str]]:
    """
    List all Slurm jobs running on a cluster by running the squeue command.

    :return: A list of dictionaries, with each dictionary representing a job
    with keys for "jobid", "partition", "name", "user", "state", "time",
    "nodes", and "nodelist_reason".
    """

    def parse_job_line(line: str) -> list[dict[str, str]]:
        """
        Parse a line of Slurm job information and return a dictionary with the job details.

        :param line: A string containing a single line of job information from the
        squeue output.
        :return: A dictionary with keys for "jobid", "partition", "name", "user",
        "state", "time", "nodes", and "nodelist_reason".
        """
        columns = ["jobid", "name", "nodelist_reason"]
        job_data = line.split()
        return {col: value for col, value in zip(columns, job_data)}

    try:
        result = subprocess.run(
            ["squeue", "--noheader", "--format=%.18i %.50j %R"], capture_output=True, text=True, check=True
        )
        jobs_output = result.stdout.strip().split("\n")

        job_list = [parse_job_line(job) for job in jobs_output if job]

        return job_list
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return []


def is_auto_eval_running(max_number) -> bool:
    """Tests if less than max_number leaderboard jobs are running."""
    jobs = list_slurm_jobs()
    leaderboard_jobs = [j["name"] for j in jobs if j["name"] == "leaderboard_backend"]
    print(f"Leaderboard jobs: {len(leaderboard_jobs)}")

    return len(leaderboard_jobs) > max_number


def check_completed_evals(
    api: HfApi,
    hf_repo: str,
    checked_status: str,
    completed_status: str,
    failed_status: str,
    local_dir: str = "./evals/",
):
    """Checks if the currently running evals are completed, if yes, update their status on the hub."""
    running_evals = get_eval_requests(checked_status, local_dir)

    for eval_request in running_evals:
        model = eval_request.model
        print("====================================")
        print(f"Checking {model}")
        slurm_jobs = list_slurm_jobs()
        job_ids = [j for j in slurm_jobs if j["jobid"] == eval_request.job_id]

        job_start_time = (
            eval_request.job_start_time.replace(":", "-") if eval_request.job_start_time is not None else ""
        )
        iso_date_format_regex = r"(-?(?:[1-9][0-9]*)?[0-9]{4})-(1[0-2]|0[1-9])-(3[01]|0[1-9]|[12][0-9])T(2[0-3]|[01][0-9])-([0-5][0-9])-([0-5][0-9])(\.[0-9]+)?(Z|[+-](?:2[0-3]|[01][0-9])-[0-5][0-9])?"

        output_file_glob = f"eval_results/results/{model}/results*.json"
        output_files = glob.glob(output_file_glob)
        output_files = [re.search(iso_date_format_regex, file) for file in output_files]
        output_files_dates = [date.group() for date in output_files if date is not None]
        output_file_exists = any([output_file_date > job_start_time for output_file_date in output_files_dates])

        if output_file_exists:
            print(
                f"EXISTS output file exists for {model} setting it to {completed_status}, jobid {eval_request.job_id}"
            )
            set_eval_request(api, eval_request, completed_status, hf_repo, local_dir)
        elif len(job_ids) == 0:
            print(
                f"No JOB ID and no result file found for {model} setting it to {failed_status}, jobid {eval_request.job_id}"
            )
            set_eval_request(api, eval_request, failed_status, hf_repo, local_dir)
        else:
            print(f"Job is still running for {model}, jobid: {eval_request.job_id}")
