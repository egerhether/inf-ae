import os
import json
import argparse
import numpy as np
from collections import defaultdict
from plot import generate_combined_cold_start_plot

def main():
    parser = argparse.ArgumentParser(
        description="Generate 3D plots with confidence intervals from multi-seed experiment results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--dataset', type=str, required=True, help="Name of the dataset.")
    parser.add_argument('--seeds', type=int, nargs='+', required=True, help="A list of random seeds used for the experiments (e.g., 41 42 43).")
    parser.add_argument('--metrics', nargs='+', required=True, help='A list of metric names to plot in subplots.')
    parser.add_argument('--results-dir', type=str, default="./results/cold-start/", help="The directory where result files are stored.")
    args = parser.parse_args()

    # Load data from all seeds
    all_metrics_data = []
    stats_data_from_one_seed = None
    
    print("Loading data from all seeds...")
    for seed in args.seeds:
        filedir = f"{args.results_dir}{args.dataset}/seed{seed}/"
        metrics_path = f"{filedir}metrics.json"
        
        with open(metrics_path, "r") as f:
            metrics_data = json.load(f)
        if metrics_data:
            all_metrics_data.append(metrics_data)
        
        # Load one stats file for the axis labels
        if stats_data_from_one_seed is None:
            stats_path = f"{filedir}stats.json"
            with open(stats_path, "r") as f:
                stats_data_from_one_seed = json.load(f)

    print("Aggregating results across seeds...")
    aggregated_values = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for single_run_metrics in all_metrics_data:
        for bin_key, coldness_dict in single_run_metrics.items():
            for coldness_key, metrics_dict in coldness_dict.items():
                for metric_name, value in metrics_dict.items():
                    aggregated_values[bin_key][coldness_key][metric_name].append(value)
    
    mean_metrics = defaultdict(lambda: defaultdict(dict))
    std_metrics = defaultdict(lambda: defaultdict(dict))

    for bin_key, coldness_dict in aggregated_values.items():
        for coldness_key, metrics_dict in coldness_dict.items():
            for metric_name, values in metrics_dict.items():
                mean_metrics[bin_key][coldness_key][metric_name] = np.mean(values)
                std_metrics[bin_key][coldness_key][metric_name] = np.std(values)

    generate_combined_cold_start_plot(
        mean_metrics=mean_metrics,
        std_metrics=std_metrics,
        stats_data=stats_data_from_one_seed,
        metrics_to_plot=args.metrics,
        dataset_name=args.dataset,
        seeds=args.seeds,
        results_dir=args.results_dir
    )
    
    print("\nDone.")

if __name__ == '__main__':
    main()