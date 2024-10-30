"""
This file uses the request parameters to construct the correct slurm command. 

This assumes that you are running on nodes with 8GPUs of 80G mem - if that is not the case, you'll need to adapt the model/data parallelism. It might also need to be adapted for the harness, as we built an API over their model launchers to launch evaluations more efficiently.
"""

import datetime
import subprocess
from dataclasses import dataclass

from transformers import AutoConfig

from .eval_requests import EvalRequest

EVAL_SCRIPT_FILE = ""  # TODO: path to the lm_eval main or lighteval main/main.py
ACCELERATE_CONFIG_FILE = (
    ""  # TODO: path to the accelerate config file/default_config.yaml - Create file running `accelerate config iirc`
)
OUTPUT_PATH = "eval_results"
LOG_SAMPLES = True
WRITE_OUT = True
PUSH_TO_HUB = True

models_that_need_trust = []


# This constructs the command to launch a job
@dataclass
class EvalJob:
    eval_script_file: str
    accelerate_config_file: str
    hub_model: str
    revision: str
    trust_remote_code: bool
    precision: str
    model_size_in_b: float
    weight_type: str
    base_model: str
    tasks: str
    output_path: str
    push_to_hub: bool
    log_samples: bool
    write_out: bool

    def build_command(self) -> str:
        """
        Builds a command to launch a slurm job for the given parameters.
        Notably checks the model size and precision for data and model parallelism.
        """
        model_args = f"pretrained={self.hub_model},revision={self.revision}"
        model_args += f",trust_remote_code={self.trust_remote_code}"
        # We can't use both a config file and args
        ## --config_file {self.accelerate_config_file}
        accelerate_args = "--multi_gpu "

        if self.precision in ["float16", "bfloat16"]:
            precision_factor = 1
            model_args += f",dtype={self.precision}"
        elif self.precision == "8bit":
            precision_factor = 2
            model_args += ",load_in_8bit=True"
        elif self.precision == "4bit":
            precision_factor = 4
            model_args += ",load_in_4bit=True"
        elif self.precision == "GPTQ":
            # A GPTQ model does not need dtype to be specified,
            # it will be inferred from the config
            config = AutoConfig.from_pretrained(self.hub_model)
            num_bits = int(config.quantization_config["bits"])
            bits_to_precision_factor = {2: 8, 3: 6, 4: 4, 8: 2}
            if num_bits in bits_to_precision_factor:
                precision_factor = bits_to_precision_factor[num_bits]
            else:
                precision_factor = 1
        else:
            raise Exception(f"Unknown precision {self.precision}.")

        if self.model_size_in_b <= 15 * precision_factor:  # DP8, MP2
            accelerate_args += "--num_processes=8 "
            model_args += ",model_parallel=True "
        elif self.model_size_in_b <= 33 * precision_factor:  # DB8
            accelerate_args += "--num_processes=8 "
            model_args += ",model_parallel=False "
        elif self.model_size_in_b <= 70 * precision_factor:  # DP4, MP2
            accelerate_args += "--num_processes=4 "
            model_args += ",model_parallel=True "
        elif self.model_size_in_b <= 140 * precision_factor:  # DP2, MP4
            accelerate_args += "--num_processes=2 "
            model_args += ",model_parallel=True "
        else:
            raise Exception(
                f"Cannot load a model bigger than {140*precision_factor}B on a single node in precision {self.precision}."
                f"Model {self.hub_model} size is {self.model_size_in_b}."
            )

        weight_type_to_arg = {
            "Original": "",
            "Delta": "--delta_weights",
            "Adapter": "--adapter_weights",
        }

        weight_type_arg = weight_type_to_arg[self.weight_type]
        base_model = "" if self.base_model == "" else f"--base_model {self.base_model}"

        return (
            f"accelerate launch {accelerate_args} "
            f"{self.eval_script_file} "
            f"--model_args {model_args} "
            f"{weight_type_arg} "
            f"{base_model} "
            f"--tasks {self.tasks} "
            f"--override_batch_size 1 "  # the above values are only sure to work with bs=1
            f"--output_path {self.output_path} "
            f"{'--log_samples' if self.log_samples else ''} "
            f"{'--write_out' if self.write_out else ''} "
            f"--hf_hub_log_args "
            f"hub_results_org=la-leaderboard,"
            f"details_repo_name=details,"
            f"results_repo_name=results,"
            f"push_results_to_hub={self.push_to_hub},"
            f"push_samples_to_hub={self.push_to_hub} "
        )


def launch_job(eval_request: EvalRequest, slurm_script_path: str, tasks_file: str) -> EvalRequest:
    """
    From a request, tests that the linked model exists, applies the delta/adapter weights if needed,
    then builds the correct command and launches the slurm job associated.
    """
    model = eval_request.model
    hub_model = model
    revision = eval_request.revision
    trust_remote_code = hub_model in models_that_need_trust

    eval_job_request = EvalJob(
        eval_script_file=EVAL_SCRIPT_FILE,
        accelerate_config_file=ACCELERATE_CONFIG_FILE,
        hub_model=hub_model,
        revision=revision,
        trust_remote_code=trust_remote_code,
        precision=eval_request.precision,
        model_size_in_b=eval_request.params,
        weight_type=eval_request.weight_type,
        base_model=eval_request.base_model,
        tasks=tasks_file,
        output_path=OUTPUT_PATH,
        log_samples=LOG_SAMPLES,
        write_out=WRITE_OUT,
        push_to_hub=PUSH_TO_HUB,
    )

    try:
        command = eval_job_request.build_command()
    except Exception as e:
        print(e)
        return eval_request

    print(f"command: {command}")

    submit_output = subprocess.check_output(["sbatch", slurm_script_path, command])
    print(submit_output.decode("utf-8"))
    job_id = submit_output.decode("utf-8").strip().split(" ")[-1]
    eval_request.job_id = job_id
    eval_request.job_start_time = datetime.datetime.now().isoformat()
    return eval_request
