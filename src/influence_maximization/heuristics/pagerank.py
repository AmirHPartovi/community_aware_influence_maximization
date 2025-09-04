from typing import List
import networkx as nx
from tqdm import trange
from src.utils.results_logger import ResultsLogger

def pagerank_heuristic(G: nx.Graph, k: int, logger: ResultsLogger = None) -> List[int]:
    """
    Select top-k nodes by PageRank.
    """
    pr = nx.pagerank(G)
    sorted_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)
    seeds = []
    for i in trange(k, desc="Selecting PageRank seeds", ncols=100):
        node, score = sorted_pr[i]
        seeds.append(node)
        if logger:
            logger.log_step(seeds, score)
    return seeds
