from basic_queue.backend.manage_requests import EvalRequest


# All the functions below sort the models in the queue based on different parameters
def sort_models_by_priority(models: list[EvalRequest]) -> list[EvalRequest]:
    private_models = [model for model in models if model.private]
    public_models = [model for model in models if not model.private]

    return sort_by_submit_date(private_models) + sort_by_submit_date(public_models)


def sort_by_submit_date(eval_requests: list[EvalRequest]) -> list[EvalRequest]:
    return sorted(eval_requests, key=lambda x: x.submitted_time, reverse=False)


def sort_by_size(eval_requests: list[EvalRequest]) -> list[EvalRequest]:
    return sorted(eval_requests, key=lambda x: x.params, reverse=False)


def sort_by_likes(eval_requests: list[EvalRequest]) -> list[EvalRequest]:
    return sorted(eval_requests, key=lambda x: x.likes, reverse=False)
