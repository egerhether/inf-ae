import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import math
import os

from utils import get_cores

def save_interaction_statistics(train_sets, val_sets, test_sets, dataset_name, seed, bin_width=2):
    """
    Generate a histogram of user interactions per data split in a single figure.
    """ 
    # Filter out empty sets and compute lengths
    train_lengths = [len(s) for s in train_sets if len(s) > 0]
    val_lengths = [len(s) for s in val_sets if len(s) > 0]
    test_lengths = [len(s) for s in test_sets if len(s) > 0]
    cores = get_cores(train_sets, val_sets, test_sets)

    num_users = len(train_lengths) + len(val_lengths) + len(test_lengths)

    # Combine all to determine global bin range
    all_lengths = train_lengths + val_lengths + test_lengths
    max_len = max(all_lengths)
    min_len = min(all_lengths)

    # Define bin edges based
    bins = np.arange(min_len, max_len + bin_width, bin_width)

    # Plot histograms
    plt.hist(train_lengths, bins=bins, alpha=0.3, label='Train', color='blue')
    plt.hist(val_lengths, bins=bins, alpha=0.3, label='Validation', color='green')
    plt.hist(test_lengths, bins=bins, alpha=0.3, label='Test', color='red')

    plt.title(f"{dataset_name.upper()} ({cores}-core) Interaction History Lengths (#users = {num_users})")
    plt.xlabel(f"Number of Interactions (bin width = {bin_width})")
    plt.ylabel("Number of Users")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save plot
    save_dir = f"./results/cold-start/{dataset_name}/seed{seed}/"
    save_path = os.path.join(save_dir, "interaction-hist.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def plot_cold_start_data(metrics_data, stats_data, metrics_to_plot, dataset_name, seed, save_dir):
    """
    Generates and saves a figure containing 3D subplots for multiple metrics.
    """
    # Set up the subplot grid
    num_metrics = len(metrics_to_plot)
    ncols = math.ceil(math.sqrt(num_metrics))
    nrows = math.ceil(num_metrics / ncols)
    
    fig, axes = plt.subplots(
        nrows=nrows, 
        ncols=ncols, 
        figsize=(ncols * 7, nrows * 6), 
        subplot_kw={'projection': '3d'}
    )
    axes = np.array(axes).flatten()

    # Loop through metrics and draw each subplot
    for i, metric in enumerate(metrics_to_plot):
        print(f"  -> Drawing subplot for {metric}...")
        _draw_single_3d_subplot(
            fig=fig,
            ax=axes[i],
            metrics_data=metrics_data,
            split_stats=stats_data,
            metric_to_plot=metric
        )


    # Hide any unused subplots
    for i in range(num_metrics, len(axes)):
        axes[i].axis('off')

    # Save the figure
    fig.suptitle(f"Cold-Start Performance on {dataset_name.upper()} (seed={seed})", fontsize=20, y = 0.95)
    fig.subplots_adjust(
        left=0.02,     
        right=0.98,    
        bottom=0.04,   
        top=0.95, 
        wspace=0.0,    
        hspace=0.15     
    )

    save_path = os.path.join(save_dir, dataset_name, f"seed{seed}", "curves.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print(f"\nSaving figure to: {save_path}")
    plt.savefig(save_path, dpi=150)
    plt.show()
    
def _draw_single_3d_subplot(fig, ax, metrics_data, split_stats, metric_to_plot):
    """
    Draws a single 3D surface plot onto a given Matplotlib subplot axis (ax).
    """
    # Parse data for labels 
    bin_keys = sorted(metrics_data.keys(), key=lambda k: int(''.join(filter(str.isdigit, k))))
    coldness_keys = sorted(metrics_data[bin_keys[0]].keys(), key=lambda k: int(''.join(filter(str.isdigit, k))))[:-1]
    
    # Prepare axes
    Z = np.zeros((len(bin_keys), len(coldness_keys)))
    for i, bin_key in enumerate(bin_keys):
        for j, coldness_key in enumerate(coldness_keys):
            metric_value = metrics_data.get(bin_key, {}).get(coldness_key, {}).get(metric_to_plot)
            if metric_value is not None:
                Z[i, j] = metric_value

    x_indices = np.arange(len(coldness_keys))
    y_indices = np.arange(len(bin_keys))
    X, Y = np.meshgrid(x_indices, y_indices)
    
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)

    # Label axes 
    ax.set_xlabel('Size of User\'s\nInput History', labelpad=15, fontsize=13)
    ax.set_ylabel('User Total\nInteraction Groups', labelpad=20, fontsize=13)
    ax.set_zlabel(metric_to_plot, fontsize=13, labelpad=5)
    
    # Label user bin ticks 
    y_tick_labels = []
    for i, bin_key in enumerate(bin_keys):
        min_val, max_val = split_stats[bin_key]['range']
        if i == 0:
            label_range = f"[{min_val}-{max_val}]"
        else:
            label_range = f"({min_val}-{max_val}]"
        y_tick_labels.append(f"{label_range}")
    ax.set_yticks(y_indices)
    ax.set_yticklabels(y_tick_labels, fontsize=10, va='center', ha='left')

    # Label user coldness ticks 
    x_tick_labels = [f"{''.join(filter(str.isdigit, k))} items" for k in coldness_keys]
    ax.set_xticks(x_indices)
    ax.set_xticklabels(x_tick_labels, fontsize=10, rotation=10)
    
    # Label metric ticks 
    ax.zaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(axis='z', labelsize=10)

    # Add color bar 
    fig.colorbar(surf, ax=ax, shrink=0.6, aspect=20, pad=0.15)

    # For better view 
    ax.view_init(elev=40, azim=-50)


def generate_combined_cold_start_plot(mean_metrics, std_metrics, stats_data, metrics_to_plot, dataset_name, seeds, results_dir):
    """
    Generates and saves a single figure containing 3D subplots for multiple metrics,
    including a visual representation of the standard deviation.
    """
    num_metrics = len(metrics_to_plot)
    ncols = math.ceil(math.sqrt(num_metrics))
    nrows = math.ceil(num_metrics / ncols)
    if num_metrics <= 4:
        ncols = num_metrics
        nrows = 1
    
    fig, axes = plt.subplots(
        nrows=nrows, 
        ncols=ncols, 
        figsize=(ncols * 7, nrows * 6), 
        subplot_kw={'projection': '3d'}
    )
    axes = np.array(axes).flatten()

    for i, metric in enumerate(metrics_to_plot):
        print(f"  -> Drawing subplot for {metric}...")
        _draw_confidence_subplot(
            fig=fig,
            ax=axes[i],
            mean_data=mean_metrics,
            std_data=std_metrics,
            stats_data=stats_data,
            metric_to_plot=metric
        )

    for i in range(num_metrics, len(axes)):
        axes[i].axis('off')

    fig.suptitle(f"Cold-Start Performance on {dataset_name.upper()} (seeds={seeds})", fontsize=20)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    save_dir = f"{results_dir}{dataset_name}"
    save_filename = f"{dataset_name}-all_seeds.png"
    save_path = os.path.join(save_dir, save_filename)
    
    print(f"\nSaving combined figure to: {save_path}")
    plt.savefig(save_path, dpi=150)
    plt.show()

def _draw_confidence_subplot(fig, ax, mean_data, std_data, stats_data, metric_to_plot):
    """
    Draws a single 3D surface plot with a confidence interval onto a subplot axis.
    """
    # Parse data and prepare grids for mean, upper, and lower bounds
    bin_keys = sorted(mean_data.keys(), key=lambda k: int(''.join(filter(str.isdigit, k))))
    coldness_keys = sorted(mean_data[bin_keys[0]].keys(), key=lambda k: int(''.join(filter(str.isdigit, k))))
    
    Z_mean = np.zeros((len(bin_keys), len(coldness_keys)))
    Z_upper = np.zeros_like(Z_mean)
    Z_lower = np.zeros_like(Z_mean)

    for i, bin_key in enumerate(bin_keys):
        for j, coldness_key in enumerate(coldness_keys):
            mean_val = mean_data.get(bin_key, {}).get(coldness_key, {}).get(metric_to_plot, 0)
            std_val = std_data.get(bin_key, {}).get(coldness_key, {}).get(metric_to_plot, 0)
            
            Z_mean[i, j] = mean_val
            Z_upper[i, j] = mean_val + std_val
            Z_lower[i, j] = mean_val - std_val

    # Prepare X and Y axes
    x_indices = np.arange(len(coldness_keys))
    y_indices = np.arange(len(bin_keys))
    X, Y = np.meshgrid(x_indices, y_indices)
    
    # Draw the three surfaces
    # MEAN surface
    ax.plot_surface(X, Y, Z_mean, cmap='viridis', edgecolor='none', alpha=0.9, antialiased=True)

    # STD surfaces
    ax.plot_surface(X, Y, Z_upper, color='red', alpha=0.4, antialiased=True)
    ax.plot_surface(X, Y, Z_lower, color='red', alpha=0.4, antialiased=True)

    ax.set_xlabel('Size of User\'s Input History', labelpad=10, fontsize=9)
    ax.set_ylabel('User Total\nInteraction Groups', labelpad=20, fontsize=9)
    ax.set_title(metric_to_plot, fontsize=12)
    
    # Y-axis tick labels (bins)
    y_tick_labels = []
    for i, bin_key in enumerate(bin_keys):
        min_val, max_val = stats_data[bin_key]['range']
        if i == 0: label_range = f"[{min_val}-{max_val}]"
        else: label_range = f"({min_val}-{max_val}]"
        y_tick_labels.append(f"\n{label_range}")
    ax.set_yticks(y_indices)
    ax.set_yticklabels(y_tick_labels, fontsize=8, va='center', ha='left')

    # X-axis tick labels (coldness)
    x_tick_labels = [f"{''.join(filter(str.isdigit, k))} items" for k in coldness_keys]
    ax.set_xticks(x_indices)
    ax.set_xticklabels(x_tick_labels, fontsize=8, rotation= 15)
    
    ax.zaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.tick_params(axis='z', labelsize=8)
    
    ax.view_init(elev=15, azim=-70)