import io
import json
import logging
import os
import time
from datetime import datetime

import datasets
import numpy as np
from lm_eval import evaluator, tasks, utils

from basic_queue.backend.manage_requests import EvalRequest
from basic_queue.envs import API, HARDWARE, TASKS_HARNESS
from basic_queue.logging import log_file, setup_logger

logging.getLogger("openai").setLevel(logging.WARNING)
logger = setup_logger(__name__)

datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True


def run_evaluation(
    eval_request: EvalRequest,
    task_names: list,
    num_fewshot: int,
    batch_size: int,
    device: str,
    local_dir: str,
    results_repo: str,
    logs_repo: str,
    leaderboard_group: str = None,
    limit: int = None,
):
    """Run one evaluation for the current evaluation request file and push the results to the hub.

    Args:
        eval_request (EvalRequest): Input evaluation request file representation.
        task_names (list): Tasks to launch.
        num_fewshot (int): Number of few shots to use.
        batch_size (int): Selected batch size.
        device (str): "cpu" or "gpu:0", depending on what you assigned to the space.
        local_dir (str): Where to save the results locally.
        results_repo (str): To which repository to upload the results.
        logs_repo (str): To which repository to upload the logs.
        limit (int, optional): Whether to use a number of samples only for the evaluation - only for debugging.

    Returns:
        dict: Evaluation results.
    """
    try:
        if limit:
            logger.info(f"LIMIT set to {limit}")
            logger.info(
                "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
            )

        task_names = get_task_names(task_names=task_names)

        if leaderboard_group:
            results, elapsed_time = evaluate_leaderboard(
                eval_request=eval_request,
                leaderboard_group=leaderboard_group,
                task_names=task_names,
                num_fewshot=num_fewshot,
                batch_size=batch_size,
                device=device,
                limit=limit,
            )
        else:
            results, elapsed_time = evaluate_tasks(
                eval_request=eval_request,
                task_names=task_names,
                num_fewshot=num_fewshot,
                batch_size=batch_size,
                device=device,
                limit=limit,
            )

        log_evaluation_results(results, elapsed_time)

        output_path = save_results_locally(results=results, eval_request=eval_request, local_dir=local_dir)

        upload_results(output_path=output_path, eval_request=eval_request, results_repo=results_repo)
        upload_logs(output_path=output_path, eval_request=eval_request, log_file_path=log_file, logs_repo=logs_repo)

        return results

    except Exception as e:
        logger.error(f"An error occurred during evaluation: {e}")
        raise


def get_task_names(task_names):
    "Get the task names."
    task_manager = tasks.TaskManager()
    all_tasks = task_manager.all_tasks
    task_names = utils.pattern_match(task_names, all_tasks)
    logger.info(f"Selected Tasks: {task_names}")
    return task_names


def evaluate_tasks(eval_request, task_names, num_fewshot, batch_size, device, limit):
    "Evaluate the tasks and log the results."

    start_time = time.time()

    results = {
        "results": {},
        "config": {
            "model_dtype": eval_request.precision,
            "model_name": eval_request.model,
            "model_sha": eval_request.revision,
            "task_names": task_names,
            "num_fewshot": num_fewshot,
            "batch_size": batch_size,
            "limit": limit,
        },
    }

    for task in task_names:
        logger.info(f"Evaluating task {task}")
        try:
            task_result = evaluator.simple_evaluate(
                model="huggingface",
                model_args=eval_request.get_model_args(),
                tasks=[task],
                num_fewshot=num_fewshot,
                batch_size=batch_size,
                device=device,
                limit=limit,
                write_out=True,
                log_samples=True,
                verbosity="DEBUG",
            )
            results["results"][task] = task_result["results"][task]
            logger.info(f"Results for task {task}: {json.dumps(task_result['results'][task], indent=2)}")
            logger.info(f"Samples:\n{task_result['samples'][task]}")
        except Exception as e:
            logger.error(f"An error occurred during evaluation of task {task}: {e}")
            continue

    end_time = time.time()
    elapsed_time = end_time - start_time

    results["results"]["time"] = elapsed_time

    return results, elapsed_time


def evaluate_leaderboard(eval_request, leaderboard_group, task_names, num_fewshot, batch_size, device, limit):
    "Evaluate the leaderboard group and log the results."

    start_time = time.time()

    results = {
        "results": {},
        "config": {
            "model_dtype": eval_request.precision,
            "model_name": eval_request.model,
            "model_sha": eval_request.revision,
            "task_names": leaderboard_group,
            "num_fewshot": num_fewshot,
            "batch_size": batch_size,
            "limit": limit,
        },
    }

    logger.info(f"Evaluating leaderboard group {leaderboard_group}")
    try:
        task_result = evaluator.simple_evaluate(
            model="huggingface",
            model_args=eval_request.get_model_args(),
            tasks=[leaderboard_group],
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            device=device,
            limit=limit,
            write_out=True,
            log_samples=True,
            verbosity="DEBUG",
        )

        logger.info(
            f"Results for leaderboard group {leaderboard_group}: {json.dumps(task_result['results'], indent=2)}"
        )

        results["results"][leaderboard_group] = task_result["results"][leaderboard_group]
        logger.info(
            f"Average for group {leaderboard_group}: {json.dumps(task_result['results'][leaderboard_group], indent=2)}"
        )
        for task in task_names:
            try:
                results["results"][task] = task_result["results"][task]
                logger.info(f"Results for task {task}: {json.dumps(task_result['results'][task], indent=2)}")
                logger.info(f"Samples:\n{task_result['samples'][task]}")
            except Exception as e:
                logger.error(f"An error occurred during evaluation of task {task}: {e}")
                continue
    except Exception as e:
        logger.error(f"An error occurred during evaluation of group {leaderboard_group}: {e}")

    end_time = time.time()
    elapsed_time = end_time - start_time

    results["results"]["time"] = elapsed_time

    return results, elapsed_time


def log_evaluation_results(results, elapsed_time):
    "Log the evaluation results and time taken."

    logger.info(f"Final results: {json.dumps(results, indent=2)}")

    missing_tasks = [task for task in TASKS_HARNESS if task not in results["results"]]
    if missing_tasks:
        logger.warning(f"Missing tasks: {missing_tasks}")
    else:
        logger.info("All tasks were successfully evaluated!")

    logger.info(
        f"Time taken to successfully evaluate {len(results['results'])}/{len(TASKS_HARNESS)} tasks using {HARDWARE}: {elapsed_time:.2f} seconds = {elapsed_time/3600:.2f} hours"
    )


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON Encoder for numpy data types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def save_results_locally(results, eval_request, local_dir):
    "Save the results locally."
    dumped = json.dumps(results, indent=2, cls=NumpyEncoder)
    logger.info(f"Dumped JSON: {dumped}")

    output_path = os.path.join(
        local_dir, *eval_request.model.split("/"), f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(dumped)

    # logger.info(f"Results table: {evaluator.make_table(results)}")
    return output_path


def upload_results(output_path, eval_request, results_repo):
    "Upload the results to the Hugging Face hub."
    try:
        logger.info(f"Uploading results file {output_path} to repository {results_repo}")
        API.upload_file(
            path_or_fileobj=output_path,
            path_in_repo=f"{eval_request.model}/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            repo_id=results_repo,
            repo_type="dataset",
        )
        logger.info("Results file uploaded successfully.")
    except Exception as e:
        logger.error(f"Failed to upload results file: {e}")
        raise


def upload_logs(output_path, eval_request, log_file_path, logs_repo):
    "Upload the logs, results and evaluated tasks to the Hugging Face hub."

    datetime_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = eval_request.model.replace("/", "_")
    run_id = f"{datetime_now}_{model_id}"

    try:
        logger.info(f"Uploading results file {output_path} to repository {logs_repo}")
        API.upload_file(
            path_or_fileobj=output_path,
            path_in_repo=f"{run_id}/{run_id}_results.json",
            repo_id=logs_repo,
            repo_type="dataset",
        )
        logger.info("Results file uploaded successfully.")
    except Exception as e:
        logger.error(f"Failed to upload results file: {e}")

    try:
        logger.info(f"Uploading logs file {log_file_path} to repository {logs_repo}")
        API.upload_file(
            path_or_fileobj=log_file_path,
            path_in_repo=f"{run_id}/{run_id}_logs.log",
            repo_id=logs_repo,
            repo_type="dataset",
        )
        logger.info("Logs file uploaded successfully.")
    except Exception as e:
        logger.error(f"Failed to upload logs file: {e}")

    try:
        # Create an in-memory file-like object for the tasks JSON
        tasks_data = json.dumps({"tasks": TASKS_HARNESS}).encode("utf-8")
        tasks_json = io.BytesIO(tasks_data)

        logger.info(f"Uploading tasks file to repository {logs_repo}")
        API.upload_file(
            path_or_fileobj=tasks_json,
            path_in_repo=f"{run_id}/{run_id}_tasks.json",
            repo_id=logs_repo,
            repo_type="dataset",
        )
        logger.info("Tasks file uploaded successfully.")
    except Exception as e:
        logger.error(f"Failed to upload tasks file: {e}")
