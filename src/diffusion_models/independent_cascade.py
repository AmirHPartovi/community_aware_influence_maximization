import networkx as nx
import random
from tqdm import tqdm  


def independent_cascade(G: nx.Graph, seeds: list, propagation_prob: float = 0.01,
                        steps: int = 0, max_steps: int = 1000, show_progress: bool = False) -> set:
    activated = set(seeds)
    newly_activated = set(seeds)
    t = 0

    # Create a progress bar for diffusion
    pbar = tqdm(total=max_steps, desc="IC diffusion",
                disable=not show_progress, ncols=80)

    while newly_activated and (steps == 0 or t < steps) and t < max_steps:
        next_activated = set()
        for node in newly_activated:
            for neighbor in G.neighbors(node):
                if neighbor not in activated and random.random() <= propagation_prob:
                    next_activated.add(neighbor)

        newly_activated = next_activated
        activated |= newly_activated
        t += 1
        pbar.update(1)

    pbar.close()
    return activated
