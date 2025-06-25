import numpy as np
import matplotlib.pyplot as plt
import json

def save_ablation_figs():

    with open("data.json", "r") as f:
        data = json.load(f)

    seeds = list(data["weak_gen"].keys())
    sizes = list(data["weak_gen"]["42"].keys())
    metrics = list(data["weak_gen"]["42"]["5k"].keys())
    sizes_vals = [int(m[:-1]) * 1000 for m in sizes]
    
    for metric in metrics:
        x_vals = sizes_vals
        colors = plt.get_cmap("viridis", 4)
    
        # Calculate mean and std for weak generalization
        weak_means = []
        weak_stds = []
        for m in sizes:
            vals = [data["weak_gen"][seed][m][metric] for seed in seeds]
            weak_means.append(np.mean(vals))
            weak_stds.append(np.std(vals))
    
        # Calculate mean and std for strong generalization
        strong_means = []
        strong_stds = []
        for m in sizes:
            vals = [data["strong_gen"][seed][m][metric] for seed in seeds]
            strong_means.append(np.mean(vals))
            strong_stds.append(np.std(vals))
    
        # Plot lines
        plt.plot(x_vals, weak_means, color=colors(0), zorder=2, label='Weak generalization', marker='o')
        plt.plot(x_vals, strong_means, color=colors(1), zorder=2, label='Strong generalization', marker='o')
    
        # Plot shaded std deviation
        plt.fill_between(x_vals, 
                         np.array(weak_means) - np.array(weak_stds), 
                         np.array(weak_means) + np.array(weak_stds), 
                         color=colors(0), alpha=0.2)
    
        plt.fill_between(x_vals, 
                         np.array(strong_means) - np.array(strong_stds), 
                         np.array(strong_means) + np.array(strong_stds), 
                         color=colors(1), alpha=0.2)
    
        # Titles and labels
        plt.title(f"{metric}: ML-20M trained on subsets")
        plt.xlabel("Training Subset Size")
        plt.ylabel(metric)
        plt.grid(True, zorder=-1)
        plt.legend()
        plt.tight_layout()

        filename = f"plots/{metric}_ablation.png"
        plt.savefig(filename)
        plt.show()

def save_paper_ablation_fig():

    with open("data.json", "r") as f:
        data = json.load(f)
    
    # Define your selection of metrics
    selected_metrics = ["precision@10", "recall@100", "ndcg@100", "mAUC:pos2"]
    
    seeds = list(data["weak_gen"].keys())
    sizes = list(data["weak_gen"]["42"].keys())
    sizes_vals = [int(m[:-1]) * 1000 for m in sizes]
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes = axes.flatten()  

    colors = plt.get_cmap("viridis", 4)
    
    for idx, metric in enumerate(selected_metrics):
        ax = axes[idx]
    
        # Compute stats for weak generalization
        weak_means = []
        weak_stds = []
        for m in sizes:
            vals = [data["weak_gen"][seed][m][metric] for seed in seeds]
            weak_means.append(np.mean(vals))
            weak_stds.append(np.std(vals))
    
        # Compute stats for strong generalization
        strong_means = []
        strong_stds = []
        for m in sizes:
            vals = [data["strong_gen"][seed][m][metric] for seed in seeds]
            strong_means.append(np.mean(vals))
            strong_stds.append(np.std(vals))
    
        # Plot weak
        ax.plot(sizes_vals, weak_means, color=colors(0), label='Weak generalization', marker='o')
        ax.fill_between(sizes_vals,
                        np.array(weak_means) - np.array(weak_stds),
                        np.array(weak_means) + np.array(weak_stds),
                        color=colors(0), alpha=0.2)
    
        # Plot strong
        ax.plot(sizes_vals, strong_means, color=colors(1), label='Strong generalization', marker='o')
        ax.fill_between(sizes_vals,
                        np.array(strong_means) - np.array(strong_stds),
                        np.array(strong_means) + np.array(strong_stds),
                        color=colors(1), alpha=0.2)
    
        ax.set_title(metric)
        ax.set_xlabel("Training Subset Size")
        ax.set_ylabel(metric)
        ax.grid(True)
    
    # Add a global legend outside subplots
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc = "center", bbox_to_anchor = (0.75, 0.95))
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for legend
    fig.suptitle("Number of Training Users Ablation on ML-20M")
    
    filename = "plots/ablation_paper_version.png"
    plt.savefig(filename)

if __name__ == "__main__":
    save_ablation_figs()
    save_paper_ablation_fig()
