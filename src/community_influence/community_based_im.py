import networkx as nx
from typing import Callable, Dict, List, Tuple


def community_based_influence_maximization(
    G: nx.Graph,
    detect_communities: Callable[[nx.Graph], Dict[int, int]],
    im_algorithm: Callable[[nx.Graph, int], List[int]],
    k: int
):
    """
    Run IM after community detection with proportional budget allocation.
    
    Returns:
        seeds_final: combined seed set across communities
        partition: dict {node: community_id}
        budget_per_comm: dict {community_id: allocated budget}
        seeds_initial: list of one representative seed per community
    """
    # --- Step 1: detect communities
    partition = detect_communities(G)

    # group nodes by community
    communities: Dict[int, List[int]] = {}
    for node, cid in partition.items():
        communities.setdefault(cid, []).append(node)

    num_communities = len(communities)

    # --- Step 2: proportional budget allocation
    total_nodes = sum(len(nodes) for nodes in communities.values())
    budget_per_comm = {
        cid: max(1, round(k * len(nodes) / total_nodes))
        for cid, nodes in communities.items()
    }

    # adjust if rounding messed up total
    allocated = sum(budget_per_comm.values())
    while allocated > k:
        max_cid = max(budget_per_comm, key=lambda x: budget_per_comm[x])
        budget_per_comm[max_cid] -= 1
        allocated -= 1
    while allocated < k:
        min_cid = min(budget_per_comm, key=lambda x: budget_per_comm[x])
        budget_per_comm[min_cid] += 1
        allocated += 1

    # --- Step 3: IM on each community
    seeds_final = []
    seeds_initial = []   # one per community
    for cid, nodes in communities.items():
        subG = G.subgraph(nodes)
        kc = budget_per_comm[cid]

        comm_seeds = im_algorithm(subG, kc)

        if comm_seeds:
            seeds_initial.append(comm_seeds[0])

        seeds_final.extend(comm_seeds)

    return seeds_final, partition, budget_per_comm, seeds_initial
