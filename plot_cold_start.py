import os
import sys
import json
import argparse

from plot import plot_cold_start_data

"""
Here is how you can run this script
python plot_cold_start.py \
    --dataset ml-1m --seed 42 \
    --metrics NDCG@10 \
    MEAN_AUC \
    TRUNCATED_RECALL@10 \
    GINI@10 \
    ENTROPY@10 \
    INTER_LIST_DISTANCE@10
"""

def load_json(path):
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at '{path}'", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate 3D plots from cold-start experiment result files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help="Name of the dataset (e.g., 'ml-1m', 'amazon_magazine')."
    )
    parser.add_argument(
        '--seed',
        type=int,
        required=True,
        help="The random seed used for the experiment."
    )
    parser.add_argument(
        '--metrics',
        nargs='+',
        required=True,
        help='A list of metric names to plot (e.g., "NDCG@10" "MEAN_AUC"). A separate plot will be generated for each.'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default="./results/cold-start/",
        help="The directory where result files are stored."
    )
    args = parser.parse_args()

    # Construct file paths from arguments
    stats_path = os.path.join(args.results_dir, args.dataset, f"seed{args.seed}", "stats.json")
    metrics_path = os.path.join(args.results_dir, args.dataset, f"seed{args.seed}", "metrics.json")

    # Load data from files
    print("Loading data...")
    metrics_data = load_json(metrics_path)
    stats_data = load_json(stats_path)
    
    plot_cold_start_data(metrics_data, stats_data, args.metrics, args.dataset, args.seed, args.results_dir)
    
    print("\nAll plots generated.")

if __name__ == '__main__':
    main()