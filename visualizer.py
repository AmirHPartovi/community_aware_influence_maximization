import src.visualizer.visualization as viz


def main():
    df = viz.load_results()

    # Basic plots
    viz.plot_spread_vs_k(df)
    viz.plot_runtime_vs_k(df)

    # Advanced analysis
    viz.plot_spread_vs_runtime(df)
    viz.plot_modularity_vs_spread(df)
    viz.plot_modularity_vs_seed_spread(df)

    # Distributions & correlations
    viz.plot_correlation_heatmap(df)
    viz.plot_metric_boxplots(df, metric="spread")
    viz.plot_metric_boxplots(df, metric="runtime_sec")

    # Budget visualization
    viz.plot_budget_per_community(df)


if __name__ == "__main__":
    main()
