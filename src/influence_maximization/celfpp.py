import heapq
from typing import List, Union
import networkx as nx
from tqdm import trange
from src.diffusion_models.estimate_spread import estimate_spread
from src.diffusion_models.independent_cascade import independent_cascade
from src.diffusion_models.linear_threshold import linear_threshold_fast as linear_threshold
from src.utils.results_logger import ResultsLogger

GraphLike = Union[nx.Graph, nx.DiGraph]


def celfpp(G: GraphLike, k: int, model="IC", propagation_prob=0.01, mc_simulations=100, logger: ResultsLogger = None) -> List[int]:
    """
    CELF++ robust:
    - robust to KeyError when using subgraphs
    - real tqdm for progress
    - Monte Carlo optimization
    """
    # only valid nodes inside the graph
    nodes = [v for v in G.nodes()]
    heap = []

    # Step 1: initial marginal gains
    print("[INFO] Computing initial marginal gains...")
    for v in trange(len(nodes), desc="Initial gains", ncols=100):
        if v not in G:
            continue
        gain = estimate_spread(G, [v], model, propagation_prob, mc_simulations)
        heapq.heappush(heap, (-gain, v, 0))

    seeds: List[int] = []
    selected_step = 1
    current_spread_cache = 0.0

    print("[INFO] Selecting seeds with CELF++...")
    for _ in trange(k, desc="Selecting seeds", ncols=100):
        while heap:
            gain, v, last_updated = heapq.heappop(heap)
            gain = -gain
            if v not in G:
                continue  # skip nodes removed from subgraph
            # Recompute gain only if outdated
            if last_updated < selected_step - 1:
                # fewer MC for intermediate steps
                new_gain = estimate_spread(G, seeds + [v], model, propagation_prob, max(mc_simulations//5, 10)) \
                    - current_spread_cache
                heapq.heappush(heap, (-new_gain, v, selected_step-1))
            else:
                seeds.append(v)
                current_spread_cache += gain
                if logger:
                    logger.log_step(seeds, gain)
                selected_step += 1
                break

    return seeds


if __name__ == "__main__":
    G = nx.erdos_renyi_graph(100, 0.05, seed=42)
    seeds = celfpp(G, k=5, model="IC", propagation_prob=0.1, mc_simulations=50)
    print("Selected seeds (CELF++):", seeds)
