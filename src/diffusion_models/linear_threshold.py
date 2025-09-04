from typing import Union, List, Set, Dict, Iterable, cast
from tqdm import tqdm
from typing import Union, List, Set, Dict, Iterable
from typing import List, Set
import networkx as nx
import random
from tqdm import tqdm  


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


# GraphLike
GraphLike = Union[nx.Graph, nx.DiGraph]


def linear_threshold_fast(
    G: GraphLike,
    seeds: List[int],
    max_steps: int = 10000,
    show_progress: bool = False
) -> Set[int]:
    """
    A fast implementation of the Linear Threshold (LT) diffusion model.
    """

    seeds = [s for s in seeds if s in G]

    thresholds: Dict[int, float] = {v: random.random() for v in G.nodes()}
    influence: Dict[int, float] = {v: 0.0 for v in G.nodes()}

    active: Set[int] = set(seeds)

    if G.is_directed():
    
        DG = cast(nx.DiGraph, G)
        deg: Dict[int, int] = dict(DG.in_degree())

        def outnbrs(u: int) -> Iterable[int]:
            return DG.successors(u)
    else:
        UG = cast(nx.Graph, G)
        deg = dict(UG.degree())

        def outnbrs(u: int) -> Iterable[int]:
            return UG.neighbors(u)
        
    for s in seeds:
        for v in outnbrs(s):
            if v not in active:
                dv = deg.get(v, 0)
                if dv > 0:
                    influence[v] += 1.0 / dv

    next_new: Set[int] = {
        v for v in G if v not in active and influence[v] >= thresholds[v]}

    pbar = tqdm(total=max_steps, desc="LT fast diffusion",
                disable=not show_progress, ncols=80)
    steps = 0

    while next_new and steps < max_steps:
        newly = next_new
        next_new = set()
        active |= newly

        for u in newly:
            for v in outnbrs(u):
                if v in active:
                    continue
                dv = deg.get(v, 0)
                if dv <= 0:
                    continue
                prev = influence[v]
                curr = prev + (1.0 / dv)
                if prev < thresholds[v] <= curr:
                    next_new.add(v)
                influence[v] = curr

        steps += 1
        pbar.update(1)

    pbar.close()
    return active
