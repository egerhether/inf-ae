import json
import argparse
import numpy as np
from collections import defaultdict
from plot import generate_combined_cold_start_plot

"""
run using:



"""

def main():
    parser = argparse.ArgumentParser(
        description="Generate 3D plots with confidence intervals from multi-seed experiment results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--dataset', type=str, required=True, help="Name of the dataset.")
    parser.add_argument('--seeds', type=int, nargs='+', required=True, help="A list of random seeds used for the experiments (e.g., 41 42 43).")
    parser.add_argument('--metrics', nargs='+', required=True, help='A list of metric names to plot in subplots.')
    parser.add_argument('--results-dir', type=str, default="./", help="The directory where result files are stored.")
    parser.add_argument('--metric_filenames', nargs='+', required=True, help="The names of the metrics.json file to plot.")
    args = parser.parse_args()

    # Load data from all seeds
    all_metrics_data = []
    stats_data_from_one_seed = None
    
    print("Loading data from all seeds and experiments...")
    for metric_filename in args.metric_filenames:
        all_metrics_data.append([])
        for seed in args.seeds:
            filedir = f"{args.results_dir}{args.dataset}/seed{seed}/"
            metrics_path = f"{filedir}{metric_filename}"
            
            with open(metrics_path, "r") as f:
                metrics_data = json.load(f)
            if metrics_data:
                all_metrics_data[-1].append(metrics_data)
            
            # Load one stats file for the axis labels
            if stats_data_from_one_seed is None:
                stats_path = f"{filedir}stats.json"
                with open(stats_path, "r") as f:
                    stats_data_from_one_seed = json.load(f)

    print("Aggregating results across seeds...")
    aggregated_values = []
    mean_metrics = []
    std_metrics = []
    metric_lims = defaultdict(lambda: { "min": float('inf'), "max": -float('inf')})

    for (i, _) in enumerate(args.metric_filenames):
        aggregated_values.append(defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
        for single_run_metrics in all_metrics_data[i]:
            for bin_key, coldness_dict in single_run_metrics.items():
                for coldness_key, metrics_dict in coldness_dict.items():
                    for metric_name, value in metrics_dict.items():
                        aggregated_values[i][bin_key][coldness_key][metric_name].append(value)
    
        mean_metrics.append(defaultdict(lambda: defaultdict(dict)))
        std_metrics.append(defaultdict(lambda: defaultdict(dict)))

        for bin_key, coldness_dict in aggregated_values[i].items():
            for coldness_key, metrics_dict in coldness_dict.items():
                for metric_name, values in metrics_dict.items():
                    metric_lims[metric_name]["min"] = min(metric_lims[metric_name]["min"], min(values))
                    metric_lims[metric_name]["max"] = max(metric_lims[metric_name]["max"], max(values))
                    mean_metrics[i][bin_key][coldness_key][metric_name] = np.mean(values)
                    std_metrics[i][bin_key][coldness_key][metric_name] = np.std(values)

    def get_section_name(filename):
        if "baseline" in filename:
            return "Original Cold-Start\nProfile"
        elif "popular_filled" in filename:
            return "Popularity-Filled\nProfile"
        else:
            return "Diversity-Filled\nProfile"

    section_names = list(map(get_section_name, args.metric_filenames))

    generate_combined_cold_start_plot(
        mean_metrics=mean_metrics,
        std_metrics=std_metrics,
        stats_data=stats_data_from_one_seed,
        metrics_to_plot=args.metrics,
        results_dir=args.results_dir,
        title=f"Cold-Start Performance on {args.dataset.upper()} (seeds={args.seeds})",
        section_names=section_names,
        dataset_name=args.dataset,
        metric_lims=metric_lims,
    )
    
    print("\nDone.")

if __name__ == '__main__':
    main()
