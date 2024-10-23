# Backend de "La Leaderboard"

- To evaluate the models in the requests dataset, run `python3 -m main_eval_basic_queue.py`
- To evaluate the combination of models and tasks in tasks_todo.json, run `python3 -m main_eval_internal_queue.py`
- To evaluate the models in the cluster's queue managed by slurm, run `python3 -m main_eval_slurm_queue.py`
