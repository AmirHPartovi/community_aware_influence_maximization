import networkx as nx


def detect_communities_label_propagation(
    G: nx.Graph,
) -> dict[int, int]:
    """
    Detect communities using Label Propagation algorithm.

    This implementation uses the Label Propagation algorithm to detect communities
    in a graph by simulating a voting process where nodes propagate their labels
    to their neighbors. The algorithm stops when no more changes occur.

    Args:
        G: Input graph

    Returns:
        dict[int, int]: Mapping from node IDs to community IDs.
    """

    communities = list(
        nx.algorithms.community.label_propagation_communities(G))
    if not communities:
        raise ValueError("No communities found")

    mapping = {}

    for cid, comm in enumerate(communities):
        for node in comm:
            mapping[node] = cid
    return mapping


if __name__ == "__main__":
    G = nx.karate_club_graph()

    print("\n Karate club graph communities (Label Propagation):")
    communities = detect_communities_label_propagation(G)
    print(communities)
