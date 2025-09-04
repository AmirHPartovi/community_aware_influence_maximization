import heapq
from typing import List, Union
import networkx as nx
from tqdm import trange
from src.diffusion_models.estimate_spread import estimate_spread
from src.diffusion_models.independent_cascade import independent_cascade
from src.diffusion_models.linear_threshold import linear_threshold_fast as linear_threshold
from src.utils.results_logger import ResultsLogger

GraphLike = Union[nx.Graph, nx.DiGraph]

def greedy_im(G: GraphLike, k: int, model="IC", propagation_prob=0.01, mc_simulations=100, logger: ResultsLogger = None) -> List[int]:
    """
    Greedy Influence Maximization with progress bar and cached spread.
    """
    seeds: List[int] = []
    current_spread_cache = 0.0

    print("[INFO] Selecting seeds with Greedy...")
    for _ in trange(k, desc="Selecting seeds", ncols=100):
        best_gain = -1
        best_node = None
        for v in G.nodes():
            if v in seeds:
                continue
            gain = estimate_spread(
                G, seeds + [v], model, propagation_prob, max(mc_simulations//5, 10)) - current_spread_cache
            if gain > best_gain:
                best_gain = gain
                best_node = v
        seeds.append(best_node)
        current_spread_cache += best_gain
        if logger:
            logger.log_step(seeds, best_gain)
    return seeds
