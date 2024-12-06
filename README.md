# Backend de "La Leaderboard"

- To evaluate the models in the requests dataset, run `python3 -m main_eval_basic_queue.py`
- To evaluate a custom combination of models and tasks, run `python3 -m main_eval_internal_queue.py` with an optional argument containing the path to the JSON file with the tasks to run which defaults to `"internal_queue/tasks_todo.json"`
- To evaluate the models in the cluster's queue managed by slurm, run `python3 -m main_eval_slurm_queue.py`

Notes:

- Check the version of the lm-evaluation-harness used in the requirements
