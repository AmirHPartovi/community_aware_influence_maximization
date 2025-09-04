import csv
import os
from datetime import datetime
import json


class ResultsLogger:
    def __init__(self, filepath="results/experiments.csv", json_path="results/experiments.json"):
        self.filepath = filepath
        self.json_path = json_path
        self.json_data = []
        os.makedirs(os.path.dirname(filepath), exist_ok=True)


        if not os.path.exists(filepath):
            with open(filepath, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "dataset",
                    "approach",      # baseline / community-aware
                    "algorithm",     # CELF, Greedy, Degree ...
                    "community_alg",  # Louvain, LabelProp, GirvanNewman, None
                    "runtime",
                    "k",
                    "modularity",
                    "spread_initial",
                    "spread_final",
                    "budget_per_community",
                    "seeds_initial",
                    "seeds_final"
                ])

    def log(self, dataset, approach, algorithm, community_alg, k, runtime, modularity=None, seeds=None, seeds_initial=None, seeds_final=None,spread=None, spread_initial=None, spread_final=None, budget_per_community=None):
        # CSV
        with open(self.filepath, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                dataset, approach, algorithm, community_alg if community_alg else "None",
                runtime, k,
                modularity if modularity else "None",
                spread_initial if spread_initial else "None", spread_final if spread_final else "None", 
                budget_per_community if budget_per_community else "None",
                seeds_initial if seeds_initial else "None", seeds_final if seeds_final else "None",
            ])
        # JSON
        self.json_data.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": dataset,
            "approach": approach,
            "algorithm": algorithm,
            "community_alg": community_alg if community_alg else None,
            "runtime": runtime,
            "k": k,
            "modularity": modularity,
            "spread_initial": spread_initial,
            "spread_final": spread_final,
            "budget_per_community": budget_per_community,
            "seeds_initial": seeds_initial,
            "seeds_final": seeds_final

        })

    def save_json(self):
        if self.json_data:
            os.makedirs(os.path.dirname(self.json_path), exist_ok=True)
            if not os.path.exists(self.json_path):
                with open(self.json_path, "w") as f:
                    json.dump(self.json_data, f, indent=4)
            else:
                with open(self.json_path, "r") as f:
                    try:
                        existing_data = json.load(f)
                    except json.JSONDecodeError:
                        existing_data = []
                existing_data.append(self.json_data)
                with open(self.json_path, "w") as f:
                    json.dump(existing_data, f, indent=4)
            print(f"[INFO] Detailed JSON results saved to {self.json_path}")

