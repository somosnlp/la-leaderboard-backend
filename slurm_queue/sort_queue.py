"""
To sort the models by first submitted first evaluated
"""

from .eval_requests import EvalRequest


def sort_models_by_priority(models: list[EvalRequest]) -> list[EvalRequest]:
    public_models = [model for model in models if not model.private]

    return sort_by_submit_date(public_models)


def sort_by_submit_date(eval_requests: list[EvalRequest]) -> list[EvalRequest]:
    return sorted(eval_requests, key=lambda x: x.submitted_time, reverse=False)
