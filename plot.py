import matplotlib.pyplot as plt
import numpy as np

from utils import get_cores

def save_interaction_statistics(train_sets, val_sets, test_sets, dataset_name, seed, bin_width=2):    
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

    # Define bin edges based on desired width
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
    plt.savefig(f"./plots/{dataset_name}-seed{seed}-interactions.png")
    plt.close()