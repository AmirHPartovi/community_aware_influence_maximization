import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict

RESULTS_CSV = "results/experiments.csv"
PLOTS_DIR = "plots"


def _ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_results():
    if not os.path.exists(RESULTS_CSV):
        raise FileNotFoundError(f"{RESULTS_CSV} not found. Run main.py first.")
    df = pd.read_csv(RESULTS_CSV)

    # standardize algorithm type column
    df["algo_type"] = df.apply(
        lambda r: "baseline" if pd.isnull(r.get("seeds_initial")) else "community-aware", axis=1
    )

    # standardize seed_selection column for compatibility with old functions
    df["seed_selection"] = df["algo_type"]

    # standardize runtime column
    if "runtime" in df.columns and "runtime_sec" not in df.columns:
        df.rename(columns={"runtime": "runtime_sec"}, inplace=True)

    # remove incomplete rows
    df = df.dropna(subset=["algo_type", "spread", "runtime_sec"])
    return df


# ----------------- BASIC PLOTS -----------------


def plot_spread_vs_k(df, dataset=None, model=None, save=True):
    _ensure_dir(PLOTS_DIR)
    sub = df.copy()
    if dataset:
        sub = sub[sub["dataset_name"] == dataset]
    if model:
        sub = sub[sub["model"] == model]

    plt.figure(figsize=(7, 5))
    for algo, group in sub.groupby("algo_type"):
        plt.plot(group["k"], group["spread"], marker="o", label=algo)

    plt.xlabel("k (number of seeds)")
    plt.ylabel("Spread")
    plt.title(
        f"Spread vs k ({dataset or 'all datasets'}, {model or 'all models'})")
    plt.legend()
    plt.grid(True)
    if save:
        fname = os.path.join(
            PLOTS_DIR, f"spread_vs_k_{dataset or 'all'}_{model or 'all'}.png")
        plt.savefig(fname, dpi=200, bbox_inches="tight")
        print(f"[PLOT] Saved {fname}")
    else:
        plt.show()
    plt.close()


def plot_runtime_vs_k(df, dataset=None, model=None, save=True):
    _ensure_dir(PLOTS_DIR)
    sub = df.copy()
    if dataset:
        sub = sub[sub["dataset_name"] == dataset]
    if model:
        sub = sub[sub["model"] == model]

    plt.figure(figsize=(7, 5))
    for algo, group in sub.groupby("algo_type"):
        plt.plot(group["k"], group["runtime_sec"], marker="s", label=algo)

    plt.xlabel("k (number of seeds)")
    plt.ylabel("Runtime (seconds)")
    plt.title(
        f"Runtime vs k ({dataset or 'all datasets'}, {model or 'all models'})")
    plt.legend()
    plt.grid(True)
    if save:
        fname = os.path.join(
            PLOTS_DIR, f"runtime_vs_k_{dataset or 'all'}_{model or 'all'}.png")
        plt.savefig(fname, dpi=200, bbox_inches="tight")
        print(f"[PLOT] Saved {fname}")
    else:
        plt.show()
    plt.close()

# ----------------- ADVANCED ANALYSIS -----------------


def plot_spread_vs_runtime(df, dataset=None, model=None, save=True):
    _ensure_dir(PLOTS_DIR)
    sub = df.copy()
    if dataset:
        sub = sub[sub["dataset_name"] == dataset]
    if model:
        sub = sub[sub["model"] == model]

    plt.figure(figsize=(7, 5))
    for algo, group in sub.groupby("algo_type"):
        plt.scatter(group["runtime_sec"], group["spread"],
                    label=algo, alpha=0.7)

    plt.xlabel("Runtime (seconds)")
    plt.ylabel("Spread")
    plt.title(
        f"Spread vs Runtime ({dataset or 'all datasets'}, {model or 'all models'})")
    plt.legend()
    plt.grid(True)
    if save:
        fname = os.path.join(
            PLOTS_DIR, f"spread_vs_runtime_{dataset or 'all'}_{model or 'all'}.png")
        plt.savefig(fname, dpi=200, bbox_inches="tight")
        print(f"[PLOT] Saved {fname}")
    else:
        plt.show()
    plt.close()


def plot_modularity_vs_spread(df, dataset=None, model=None, save=True):
    _ensure_dir(PLOTS_DIR)
    sub = df.copy()
    if dataset:
        sub = sub[sub["dataset_name"] == dataset]
    if model:
        sub = sub[sub["model"] == model]

    plt.figure(figsize=(7, 5))
    for algo, group in sub.groupby("algo_type"):
        plt.scatter(group["modularity"], group["spread"],
                    label=algo, alpha=0.7)

    plt.xlabel("Modularity")
    plt.ylabel("Spread")
    plt.title(
        f"Modularity vs Spread ({dataset or 'all datasets'}, {model or 'all models'})")
    plt.legend()
    plt.grid(True)
    if save:
        fname = os.path.join(
            PLOTS_DIR, f"modularity_vs_spread_{dataset or 'all'}_{model or 'all'}.png")
        plt.savefig(fname, dpi=200, bbox_inches="tight")
        print(f"[PLOT] Saved {fname}")
    else:
        plt.show()
    plt.close()


def plot_modularity_vs_seed_spread(df, dataset=None, model=None, save=True):
    _ensure_dir(PLOTS_DIR)
    sub = df.copy()
    if dataset:
        sub = sub[sub["dataset_name"] == dataset]
    if model:
        sub = sub[sub["model"] == model]

    plt.figure(figsize=(7, 5))
    for algo, group in sub.groupby("algo_type"):
        if "spread_initial" in group.columns and "spread_final" in group.columns:
            plt.scatter(group["modularity"], group["spread_initial"],
                        label=f"{algo} (initial)", alpha=0.6, marker="o")
            plt.scatter(group["modularity"], group["spread_final"],
                        label=f"{algo} (final)", alpha=0.6, marker="s")
        else:
            plt.scatter(group["modularity"], group["spread"],
                        label=algo, alpha=0.7, marker="d")

    plt.xlabel("Modularity")
    plt.ylabel("Seed Spread")
    plt.title(
        f"Modularity vs Seed Spread ({dataset or 'all datasets'}, {model or 'all models'})")
    plt.legend()
    plt.grid(True)
    if save:
        fname = os.path.join(
            PLOTS_DIR, f"modularity_vs_seed_spread_{dataset or 'all'}_{model or 'all'}.png")
        plt.savefig(fname, dpi=200, bbox_inches="tight")
        print(f"[PLOT] Saved {fname}")
    else:
        plt.show()
    plt.close()

# ----------------- DISTRIBUTIONS & CORRELATIONS -----------------


def plot_correlation_heatmap(df, save=True):
    _ensure_dir(PLOTS_DIR)
    numeric_df = df.select_dtypes(include=["float64", "int64"])
    corr = numeric_df.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap of Metrics")
    if save:
        fname = os.path.join(PLOTS_DIR, "correlation_heatmap.png")
        plt.savefig(fname, dpi=200, bbox_inches="tight")
        print(f"[PLOT] Saved {fname}")
    else:
        plt.show()
    plt.close()


def plot_metric_boxplots(df, metric="spread", save=True):
    _ensure_dir(PLOTS_DIR)
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x="algo_type", y=metric)
    sns.stripplot(data=df, x="algo_type", y=metric, color="black", alpha=0.4)
    plt.title(f"Distribution of {metric} across Algorithms")
    plt.xticks(rotation=45)
    if save:
        fname = os.path.join(PLOTS_DIR, f"boxplot_{metric}.png")
        plt.savefig(fname, dpi=200, bbox_inches="tight")
        print(f"[PLOT] Saved {fname}")
    else:
        plt.show()
    plt.close()

# ----------------- BUDGET PER COMMUNITY -----------------


def plot_budget_per_community(df, save=True):
    if "budget_per_community" not in df.columns:
        print("[WARN] budget_per_community column not found.")
        return

    _ensure_dir(PLOTS_DIR)
    # فرض: budget_per_community به صورت dict یا string JSON ذخیره شده
    df_copy = df.copy()
    df_copy["budget_per_community"] = df_copy["budget_per_community"].apply(
        lambda x: eval(x) if isinstance(x, str) else x
    )

    community_budget: Dict[int, int] = {}
    for budgets in df_copy["budget_per_community"].dropna():
        for comm, val in budgets.items():
            community_budget[comm] = community_budget.get(comm, 0) + val

    plt.figure(figsize=(8, 6))
    plt.bar(community_budget.keys(), community_budget.values())
    plt.xlabel("Community")
    plt.ylabel("Total Budget")
    plt.title("Budget Allocation per Community")
    if save:
        fname = os.path.join(PLOTS_DIR, "budget_per_community.png")
        plt.savefig(fname, dpi=200, bbox_inches="tight")
        print(f"[PLOT] Saved {fname}")
    else:
        plt.show()
    plt.close()
