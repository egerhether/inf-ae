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

TITLE_SIZE=35
SUBTITLE_SIZE=28
AX_LABEL_SIZE=23
AX_VAL_SIZE=14

def generate_combined_cold_start_plot(mean_metrics, std_metrics, stats_data, metrics_to_plot,
        results_dir, title, section_names, dataset_name, metric_lims):
    """
    Generates and saves a single figure containing 3D subplots for multiple metrics,
    including a visual representation of the standard deviation.
    """
    num_metrics = len(metrics_to_plot)
    ncols = math.ceil(math.sqrt(num_metrics))
    nrows = math.ceil(num_metrics / ncols) * len(section_names)
    if num_metrics <= 4:
        ncols = num_metrics
        nrows = 1 * len(section_names)
    
    fig, axes = plt.subplots(
        nrows=nrows, 
        ncols=ncols, 
        figsize=(ncols * 7, nrows * 6), 
        subplot_kw={'projection': '3d'}
    )
    axes = np.array(axes).flatten()

    ax_pos=0
    for (j, section_name) in enumerate(section_names):
        for i, metric in enumerate(metrics_to_plot):
            print(f"  -> Drawing subplot for {metric}...")
            _draw_confidence_subplot(
                fig=fig,
                ax=axes[ax_pos],
                mean_data=mean_metrics[j],
                std_data=std_metrics[j],
                stats_data=stats_data,
                metric_to_plot=metric,
                section_name=section_name if i == 0 else None,
                row=j+1,
                num_rows=len(section_names),
                lims=metric_lims[metric]
            )
            ax_pos+=1

    fig.suptitle(title, fontsize=TITLE_SIZE, y=0.98)

    fig.subplots_adjust(
        left=0.08,
        right=0.9,
        bottom=0.1,
        top=0.9,
        wspace=0.1,
        hspace=0.2
    )

    save_dir = f"{results_dir}{dataset_name}"
    save_filename = f"{dataset_name}-plot.png"
    save_path = os.path.join(save_dir, save_filename)
    
    print(f"\nSaving combined figure to: {save_path}")
    plt.savefig(save_path, dpi=150)
    plt.show()

def _draw_confidence_subplot(fig, ax, mean_data, std_data, stats_data, metric_to_plot, section_name, row, num_rows, lims):
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
            mean_val = mean_data.get(bin_key).get(coldness_key).get(metric_to_plot)
            std_val = std_data.get(bin_key).get(coldness_key).get(metric_to_plot)
            
            Z_mean[i, j] = mean_val
            Z_upper[i, j] = mean_val + std_val
            Z_lower[i, j] = mean_val - std_val

    assert np.all(Z_lower <= Z_upper)

    # Prepare X and Y axes
    x_indices = np.arange(len(coldness_keys))
    y_indices = np.arange(len(bin_keys))
    X, Y = np.meshgrid(x_indices, y_indices)
    
    # Draw the three surfaces
    # STD bottom surfaces
    ax.plot_surface(X, Y, Z_lower, color='#ff2929',edgecolor='#ff2929', alpha=0.2, antialiased=True, zorder=1)

    # STD upper surfaces
    ax.plot_surface(X, Y, Z_upper, color='#ff2929', edgecolor='#ff2929', alpha=0.2, antialiased=True, zorder=1)

    # MEAN surface
    ax.plot_surface(X, Y, Z_mean, cmap='viridis', alpha=0.9, antialiased=True, zorder=2)

    ax.set_xlabel('Input Items', labelpad=20, fontsize=AX_LABEL_SIZE)
    ax.set_ylabel('History Length', labelpad=20, fontsize=AX_LABEL_SIZE)
    if row == 1: 
        print(metric_to_plot)
        if metric_to_plot == "MEAN_AUC":
            ax.set_title("mAUC:pos2", fontsize=SUBTITLE_SIZE)
        elif metric_to_plot == "GLOBAL_ENTROPY@10":
            ax.set_title("gEntropy@10", fontsize=SUBTITLE_SIZE)
        else:
            ax.set_title(metric_to_plot, fontsize=SUBTITLE_SIZE)
    
    # Y-axis tick labels (bins)
    y_tick_labels = []
    for i, bin_key in enumerate(bin_keys):
        min_val, max_val = stats_data[bin_key]['range']
        if i == 0: label_range = f"[{min_val}-{max_val}]"
        else: label_range = f"({min_val}-{max_val}]"
        y_tick_labels.append(f"\n{label_range}")
    ax.set_yticks(y_indices)
    ax.set_yticklabels(y_tick_labels, fontsize=AX_VAL_SIZE, va='center', ha='left', rotation=-5)

    # X-axis tick labels (coldness)
    x_tick_labels = [f"{''.join(filter(str.isdigit, k))}" for k in coldness_keys]
    ax.set_xticks(x_indices)
    ax.set_xticklabels(x_tick_labels, fontsize=AX_VAL_SIZE, va='center', rotation=-5)
    
    ax.zaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(axis='z', labelsize=AX_VAL_SIZE)
    ax.set_zlim(lims["min"], lims["max"])
    
    ax.view_init(elev=15, azim=-45)

    if section_name is not None:
        # Add section name to the left of the plot, rotated vertically
        adjustment = 0.2 if row == 1 else 0.6 if row == num_rows else 0.4
        fig.text(0.06, 1 - (row - adjustment) / num_rows, section_name, rotation='vertical', fontsize=SUBTITLE_SIZE, va='center', ha='center')