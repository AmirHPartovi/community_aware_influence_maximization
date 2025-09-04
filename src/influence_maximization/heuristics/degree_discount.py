from typing import List
import networkx as nx
from tqdm import trange
from src.utils.results_logger import ResultsLogger
def degree_discount_heuristic(G: nx.Graph, k: int, logger: ResultsLogger = None) -> List[int]:
    """
    Degree Discount heuristic for Influence Maximization.
    """
    degrees = {v: d for v, d in G.degree()}
    t = {v: 0 for v in G.nodes()}
    seeds = []

    for _ in trange(k, desc="Selecting Degree Discount seeds", ncols=100):
        u = max(degrees, key=degrees.get)
        seeds.append(u)
        if logger:
            logger.log_step(seeds, degrees[u])
        degrees[u] = -1  # mark selected
        for v in G.neighbors(u):
            if degrees[v] != -1:
                t[v] += 1
                degrees[v] = G.degree(
                    v) - 2 * t[v] - (G.degree(v) - t[v]) * t[v] / max(1, G.degree(v))
    return seeds



if __name__ == "__main__":
    G = nx.erdos_renyi_graph(20, 0.2, seed=42)
    seeds = degree_discount_heuristic(G, 5)
    print("Selected seeds (Degree Discount Heuristic):", seeds)
