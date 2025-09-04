
from src.diffusion_models.independent_cascade import independent_cascade
from src.diffusion_models.linear_threshold import linear_threshold
from typing import List, Set, Dict, Tuple
import networkx as nx
from .independent_cascade import independent_cascade

from src.diffusion_models.linear_threshold import linear_threshold_fast


CacheKey = Tuple[str, float, Tuple[int, ...]]


def estimate_spread(
    G: nx.Graph,
    seeds: List[int],
    model: str = "IC",
    propagation_prob: float = 0.01,
    mc_simulations: int = 100,
) -> float:
    """Estimate the spread of influence in a network."""
    total = 0
    for _ in range(mc_simulations):
        if model == "IC":
            activated = independent_cascade(G, seeds, propagation_prob)
        elif model == "LT":
            activated = linear_threshold(G, seeds)
        else:
            raise ValueError("model must be 'IC' or 'LT'")
        total += len(activated)
    return total / mc_simulations


def estimate_spread_cached(
    G: nx.Graph,
    seeds: List[int],
    model: str = "IC",
    propagation_prob: float = 0.01,
    mc_simulations: int = 100,
    cache: Dict[CacheKey, float] | None = None,
) -> float:
    """Estimate the spread of influence in a network with caching."""
    if cache is None:
        return estimate_spread(G, seeds, model, propagation_prob, mc_simulations)

    key: CacheKey = (model, propagation_prob, tuple(sorted(seeds)))
    if key in cache:
        return cache[key]

    val = estimate_spread(G, seeds, model, propagation_prob, mc_simulations)
    cache[key] = val
    return val
