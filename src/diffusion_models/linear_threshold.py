import numpy as np
import random
from typing import Union, List, Set, Dict, Iterable, cast
from tqdm import tqdm  
import networkx as nx


def linear_threshold(G: nx.Graph, seeds: list, steps: int = 0,
                     max_steps: int = 1000, show_progress: bool = False) -> set:
    thresholds = {node: random.random() for node in G.nodes()}
    activated = set(seeds)
    newly_activated = set(seeds)
    t = 0

    pbar = tqdm(total=max_steps, desc="LT diffusion",
                disable=not show_progress, ncols=80)

    while newly_activated and (steps == 0 or t < steps) and t < max_steps:
        next_activated = set()
        for node in G.nodes():
            if node not in activated:
                neighbors = list(G.neighbors(node))
                if not neighbors:
                    continue
                active_neighbors = sum(1 for n in neighbors if n in activated)
                influence = active_neighbors / len(neighbors)
                if influence >= thresholds[node]:
                    next_activated.add(node)

        newly_activated = next_activated
        activated |= newly_activated
        t += 1
        pbar.update(1)

    pbar.close()
    return activated


# src/diffusion_models/linear_threshold.py

GraphLike = Union[nx.Graph, nx.DiGraph, nx.MultiDiGraph, nx.MultiGraph]


def linear_threshold_fast(G: nx.Graph, seeds: List[int], max_steps: int = 10000, show_progress: bool = False) -> Set[int]:
    """
    Optimized Linear Threshold diffusion model (vectorized).
    """
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # adjacency list (as numpy indices)
    neighbors = [np.array([node_to_idx[v]
                          for v in G.neighbors(u)], dtype=np.int32) for u in nodes]

    # degrees
    deg = np.array([len(neigh) for neigh in neighbors], dtype=np.int32)

    # random thresholds [0,1)
    thresholds = np.random.rand(n)

    # state
    influence = np.zeros(n, dtype=np.float32)
    active = np.zeros(n, dtype=np.bool_)
    newly = np.zeros(n, dtype=np.bool_)

    # initialize seeds
    for s in seeds:
        idx = node_to_idx[s]
        active[idx] = True
        newly[idx] = True

    steps = 0
    while steps < max_steps and newly.any():
        current_new = np.where(newly)[0]
        newly[:] = False

        for u in current_new:
            for v in neighbors[u]:
                if active[v] or deg[v] == 0:
                    continue
                influence[v] += 1.0 / deg[v]
                if influence[v] >= thresholds[v]:
                    newly[v] = True

        active |= newly
        steps += 1

    return {nodes[i] for i, flag in enumerate(active) if flag}
