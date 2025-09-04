"""
Main entry for Influence Maximization experiments:
- Baseline IM (no community detection)
- Community-aware IM (run IM after community detection)

Project expects:
  data/                 # SNAP datasets
  results/              # logs & plots
  src/                  # all modules (as designed together)

Run examples:
  python3 main.py --dataset facebook_combined.txt --approach both --algo celfpp --model IC --k 50 --save-json
  python3 main.py --dataset facebook_combined.txt --approach community --community louvain --algo celfpp --model IC --k 50 --plot --save-json
  python3 main.py --dataset facebook_combined.txt --approach both --community louvain --algo celfpp --model IC --k 50  --save-json
  python3 main.py --dataset wiki-Vote.txt --approach both --community louvain --algo celfpp --model IC --k 50  --save-json
  python3 main.py --dataset soc-Epinions1.txt --approach community --community louvain --algo celfpp --model IC --k 50  --save-json
  python3 main.py --dataset soc-sign-bitcoinotc.csv --approach both --community louvain --algo celfpp --model IC --k 50  --save-json
  python3 main.py --dataset facebook_combined.txt --approach both --community louvain --algo celfpp --model IC --k 50 --plot --save-json

"""

import os
import time
import random
import argparse
import networkx as nx


# ---- imports from our src/ tree ----
from src.evaluation.influence_eval import evaluate_spread, evaluate_runtime
from src.evaluation.community_eval import evaluate_modularity
from src.utils.results_logger import ResultsLogger

# diffusion models are called indirectly via evaluate_spread (IC / LT)
from src.diffusion_models.independent_cascade import independent_cascade
from src.diffusion_models.linear_threshold import linear_threshold

# seed selection algorithms
from src.influence_maximization.greedy import greedy_im
from src.influence_maximization.celf import celf
from src.influence_maximization.celfpp import celfpp
from src.influence_maximization.heuristics.degree_discount import degree_discount_heuristic
from src.influence_maximization.heuristics.degree import degree_heuristic
from src.influence_maximization.heuristics.pagerank import pagerank_heuristic

# community detection
from src.community_detection.louvain import detect_communities_louvain
from src.community_detection.label_propagation import detect_communities_label_propagation
from src.community_detection.girvan_newman import detect_communities_girvan_newman

# community-aware IM helper
from src.community_influence.community_based_im import community_based_influence_maximization



def load_graph_from_file(path: str, directed: bool = False) -> nx.Graph:
    """
    Generic loader for SNAP-like edge lists.
    Handles:
    - Space-delimited edge lists (.txt)
    - CSV files with different formats including Bitcoin OTC dataset
    Returns an undirected Graph by default (set directed=True for DiGraph).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
        
    Gtype = nx.DiGraph if directed else nx.Graph
    
    if os.path.splitext(path)[1] == ".txt":
        try:
            G = nx.read_edgelist(path, comments="#", 
                               nodetype=int, create_using=Gtype())
            if G.number_of_edges() > 0:
                return G
        except Exception:
            pass

    if os.path.splitext(path)[1] == ".csv":
        try:
            # First try simple edge list format
            G = nx.read_edgelist(path, delimiter=",", comments="#",
                               nodetype=int, create_using=Gtype())
            if G.number_of_edges() > 0:
                return G
        except Exception:
            # Handle Bitcoin OTC format: source,target,rating,timestamp
            G = Gtype()
            with open(path, 'r') as f:
                for line in f:
                    try:
                        values = line.strip().split(',')
                        if len(values) >= 2:
                            u, v = int(values[0]), int(values[1])
                            # Add edge with weight (rating) if available
                            if len(values) >= 3:
                                weight = float(values[2])
                                G.add_edge(u, v, weight=weight)
                            else:
                                G.add_edge(u, v)
                    except ValueError:
                        print(f"Skipping invalid line: {line.strip()}")
            if G.number_of_edges() > 0:
                return G
                
    raise RuntimeError(f"Could not parse dataset: {path}")


def pick_seed_set(G: nx.Graph, method: str, k: int, model: str, p: float, mc: int):
    """
    Dispatch seed selection based on method string.
    """
    method = method.lower()
    if method == "random":
        return random.sample(list(G.nodes()), k)
    if method == "degree":
        return degree_heuristic(G, k)
    if method == "degree_discount":
        return degree_discount_heuristic(G, k)
    if method == "pagerank":
        return pagerank_heuristic(G, k)
    if method == "greedy":
        return greedy_im(G, k, model=model, propagation_prob=p, mc_simulations=mc)
    if method == "celf":
        return celf(G, k, model=model, propagation_prob=p, mc_simulations=mc)
    if method == "celfpp":
        return celfpp(G, k, model=model, propagation_prob=p, mc_simulations=mc)
    raise ValueError(f"Unknown seed selection method: {method}")


def get_community_detector(name: str):
    name = (name or "").lower()
    if name in ("louvain", "lv"):
        return detect_communities_louvain, "Louvain"
    if name in ("label_prop", "label-prop", "labelprop", "lp"):
        return detect_communities_label_propagation, "LabelPropagation"
    if name in ("girvan_newman", "girvan-newman", "gn"):
        # We'll wrap to allow level param via argparse later if you want to extend
        def _gn(G: nx.Graph):
            return detect_communities_girvan_newman(G, level=1)
        return _gn, "GirvanNewman"
    raise ValueError(f"Unknown community algorithm: {name}")



def run_baseline(G: nx.Graph, args, logger: ResultsLogger, dataset_label: str):
    """
    Run baseline IM (no community detection).
    Logs results to JSON/CSV
    """
    # الگوریتم انتخاب seed
    def im_algo_wrapper(G: nx.Graph, k: int):
        return pick_seed_set(G, args.algo, k, args.model, args.p, args.mc)

    t0 = time.time()
    seeds_final = im_algo_wrapper(G, args.k)

    # diffusion: initial (یک seed اول) + final (همه)
    seeds_initial = seeds_final[:1] if seeds_final else []
    activated_initial = evaluate_spread(G, seeds_initial, model=args.model)
    activated_final = evaluate_spread(G, seeds_final, model=args.model)

    t1 = time.time()
    spread_initial = len(activated_initial)
    spread_final = len(activated_final)
    runtime = t1 - t0

    # cause don't have communities, modularity = 0
    modularity = 0.0

    logger.log(
        dataset=dataset_label,
        approach="baseline",
        algorithm=args.algo.upper(),
        community_alg=None,
        runtime=round(runtime, 6),
        k=args.k,
        modularity=round(modularity, 6),
        spread_initial=spread_initial,
        spread_final=spread_final,
        seeds_initial=seeds_initial,
        seeds_final=seeds_final
    )

    print(
        f"[BASELINE] algo={args.algo} | seeds_init={seeds_initial} | seeds_final={seeds_final[:10]}{'...' if len(seeds_final)>10 else ''}")
    print(
        f"[BASELINE] spread_init={spread_initial} | spread_final={spread_final} | time={runtime:.4f}s")

    return spread_final, runtime, seeds_final, modularity

def run_community_aware(G: nx.Graph, args, logger: ResultsLogger, dataset_label: str):
    """
    Run community detection, then IM inside communities (proportional budget split).
    Logs results to JSON/CSV
    """
    detect_fn, comm_name = get_community_detector(args.community)

    def im_algo_wrapper(subG: nx.Graph, kc: int):
        return pick_seed_set(subG, args.algo, kc, args.model, args.p, args.mc)

    t0 = time.time()
    seeds_final, partition, budget_per_comm, seeds_initial = community_based_influence_maximization(
        G,
        detect_communities=detect_fn,
        im_algorithm=im_algo_wrapper,
        k=args.k
    )

    # diffusion for initial and final
    activated_initial = evaluate_spread(G, seeds_initial, model=args.model)
    activated_final = evaluate_spread(G, seeds_final, model=args.model)

    t1 = time.time()

    spread_initial = len(activated_initial)
    spread_final = len(activated_final)
    runtime = t1 - t0
    modularity = evaluate_modularity(G, partition)

    logger.log(
        dataset=dataset_label,
        approach="community-aware",
        algorithm=args.algo.upper(),
        community_alg=comm_name,
        runtime=round(runtime, 6),
        k=args.k,
        modularity=round(modularity, 6),
        seeds_initial=seeds_initial,
        seeds_final=seeds_final,
        spread_initial=spread_initial,
        spread_final=spread_final,
    )

    print(f"[COMMUNITY] comm={comm_name} | algo={args.algo} | seeds_init={seeds_initial} | seeds_final={seeds_final[:10]}{'...' if len(seeds_final)>10 else ''}")
    print(
        f"[COMMUNITY] spread_init={spread_initial} | spread_final={spread_final} | modularity={modularity:.4f} | time={runtime:.4f}s")

    return spread_final, runtime, seeds_final, modularity




def parse_args():
    p = argparse.ArgumentParser(
        description="Influence Maximization Experiments")
    p.add_argument("--data-dir", default="data",
                   help="Directory containing datasets")
    p.add_argument("--dataset", required=True,
                   help="Dataset filename, e.g., facebook_combined.txt")
    p.add_argument("--directed", action="store_true",
                   help="Treat graph as directed (default undirected)")

    p.add_argument("--approach", choices=["baseline", "community", "both"], default="both",
                   help="Run baseline IM, community-aware IM, or both")
    p.add_argument("--community", default="louvain",
                   help="Community algorithm: louvain | label_prop | girvan_newman (used if approach includes community)")
    p.add_argument("--algo", default="celfpp",
                   help="Seed selection method: random | degree | degree_discount | pagerank | greedy | celf | celfpp")

    p.add_argument("--model", choices=["IC", "LT"],
                   default="LT", help="Diffusion model")
    p.add_argument("--k", type=int, default=10, help="Seed budget")
    p.add_argument("--p", type=float, default=0.01,
                   help="Propagation prob (IC only)")
    p.add_argument("--mc", type=int, default=50,
                   help="MC sims for greedy/celf/celfpp")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")
    p.add_argument("--plot", action="store_true",
                   help="Generate plots after experiment")
    p.add_argument("--save-json", action="store_true",
               help="Save detailed results to JSON")


    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    dataset_path = os.path.join(args.data_dir, args.dataset)
    G = load_graph_from_file(dataset_path, directed=args.directed)
    dataset_label = os.path.basename(dataset_path)

    print(
        f"[INFO] Loaded {dataset_label} |V|={G.number_of_nodes()} |E|={G.number_of_edges()} | directed={args.directed}")
    print(f"[INFO] approach={args.approach} | community={args.community if args.approach!='baseline' else '-'} | algo={args.algo} | model={args.model} | k={args.k}")

    logger = ResultsLogger(filepath="results/experiments.csv")

    if args.approach in ("baseline", "both"):
        run_baseline(G, args, logger, dataset_label)

    if args.approach in ("community", "both"):
        run_community_aware(G, args, logger, dataset_label)
    
    if args.save_json:
        logger.save_json()

    if args.plot:
        import visualizer
        visualizer.main()


    print("[INFO] Results saved to results/experiments.csv")


if __name__ == "__main__":
    main()
