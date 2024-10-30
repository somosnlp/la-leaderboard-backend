#!/bin/bash
#Define your SBATCH parameters on the number of nodes, job name, etc

export TMPDIR=<YOUR TMP DIR>
export HF_HUB_ENABLE_HF_TRANSFER=1

export HF_HOME=<YOUR TMP DIR>
export HUGGINGFACE_HUB_CACHE=<YOUR TMP DIR>
export HF_DATASETS_CACHE=<YOUR TMP DIR>
export HF_MODULES_CACHE=<YOUR TMP DIR>

export HUGGING_FACE_HUB_TOKEN=<your token>

source /<... path to your conda>/conda.sh
conda activate <your conda env>

srun $1
