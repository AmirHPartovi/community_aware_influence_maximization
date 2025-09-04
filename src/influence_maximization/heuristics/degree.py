from typing import List
import networkx as nx
from tqdm import trange
from src.utils.results_logger import ResultsLogger


def degree_heuristic(G: nx.Graph, k: int, logger: ResultsLogger = None) -> List[int]:
    """
    Select top-k nodes with highest degree.
    """
    degrees = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    seeds = []
    for i in trange(k, desc="Selecting Degree seeds", ncols=100):
        node = degrees[i][0]
        seeds.append(node)
        if logger:
            logger.log_step(seeds, degrees[i][1])
    return seeds
