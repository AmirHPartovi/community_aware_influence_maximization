import time
from src.diffusion_models.independent_cascade import independent_cascade
from src.diffusion_models.linear_threshold import linear_threshold


def evaluate_spread(G, seeds, model="IC"):
    """
    run diffusion model and calculate  spread
    """
    if model == "IC":
        return independent_cascade(G, seeds)
    elif model == "LT":
        return linear_threshold(G, seeds)
    else:
        raise ValueError(f"Unknown diffusion model: {model}")


def evaluate_runtime(algorithm_func, *args, **kwargs):
    """
        runtime calculator
    """
    start = time.time()
    result = algorithm_func(*args, **kwargs)
    runtime = time.time() - start
    return result, runtime
