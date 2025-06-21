import jax
import numpy as np
from scipy.sparse import vstack, csr_matrix
from collections import defaultdict
from pprint import pformat

import eval_metrics
from data import Dataset
from utils import get_cores

def run_cold_start_experiment(
        data: Dataset,
        hyper_params,
        krr_forward,
        train_matrix,
        precomputed_alpha,
        k_values = [10, 100]
    ):

    cold_start_splits, cold_start_stats = prepare_cold_start_data(
        data.data["test_positive_set"],
        data.data["test_matrix"],
        get_cores(
            data.data["train_positive_set"],
            data.data["val_positive_set"],
            data.data["test_positive_set"]
        ),
        hyper_params["simulated_max_interactins"],
        hyper_params["cold_start_bins"],
        hyper_params["simulated_coldness_levels"],
    )

    max_k = max(k_values)
    metrics = {}
    for (bin_key, coldness_splits) in cold_start_splits.items():
        metrics[bin_key] = {}
        for (coldness_key, split) in coldness_splits.items():
            metrics[bin_key][coldness_key] = {}

            logits = krr_forward(
                X_train=train_matrix,
                X_predict=split["input_matrix"].todense(),
                reg=hyper_params["lamda"],
                alpha=precomputed_alpha,
            )

            recommendations = get_recommendations(logits, split["input_items"], max_k)

            for k in k_values:

                metrics_k = defaultdict(float)
                for i in range(len(recommendations)):
                    metrics_k[f"PRECISION@{k}"] += eval_metrics.precision(recommendations[i], split["ground_truth_items"][i], k)
                    metrics_k[f"RECALL@{k}"] += eval_metrics.recall(recommendations[i], split["ground_truth_items"][i], k)
                    metrics_k[f"NDCG@{k}"] += eval_metrics.ndcg(recommendations[i], split["ground_truth_items"][i], k)
                
                num_users = cold_start_stats[bin_key]["#users"][coldness_key]
                for metric in metrics_k:
                    metrics_k[metric] /= num_users

                metrics[bin_key][coldness_key] = metrics_k

    # with open(f"./results/cold-start/{hyper_params['dataset']}-{hyper_params['seed']}.splits.txt", "w") as f:
    #     f.write(pformat(cold_start_splits, indent=4))
    # with open(f"./results/cold-start/{hyper_params['dataset']}-{hyper_params['seed']}.stats.txt", "w") as f:
    #     f.write(pformat(cold_start_stats, indent=4))
    with open(f"./results/cold-start/{hyper_params['dataset']}-{hyper_params['seed']}.metrics.txt", "w") as f:
        f.write(pformat(metrics, indent=4))
    
def prepare_cold_start_data(
    test_positive_set: list[set[int]],
    test_matrix: csr_matrix,
    min_interactions: int,
    max_interactions: int,
    interaction_bins: int,
    simulated_coldness_levels: list[int],
):
    """
    - Splits users into bins based on their number of interactions.
    - Each user history is used to simulate a cold-start scenario where 
      a small part of the history is used as input, and the rest for evaluation.
    - **Assumes** strong generalization. <-> Positive set = full interaction history.

    Args:
        test_positive_set: List of sets of item ids representing user-items histories.
        test_matrix: The sparse interaction matrix for the test set. Each row is a user.
        min_interactions: Lower bound for the first interaction bin.
        max_interactions: Upper bound for the last interaction bin. After the initial
            bins are determined, another bin is created for all other users with higher
            number of interactions (rest-of-the-users bin). 
        interaction_bins: Number of bins to create between min and max interactions.
        simulated_coldness_levels: A list of integers, where each integer represents 
            a number of interactions to be used as input for the model, simulating a
            user with that level of "coldness".

    Returns:
        A dictionary structured for evaluating cold-start performance. The structure is:
        {
            "bin<bin_id>": {
                "coldness<#interactions>": {
                    "users": [user_id_1, user_id_2, ...],
                    "input_matrix": <scipy.sparse.matrix of shape [#users, #items]>,
                    "input_items": [[items_ids...], [items_ids...], ...],
                    "ground_truth_items": [[items_ids...], [items_ids...], ...],
                }, 
                ...
            }, 
            ...
        }

        And a bin statistics list which looks like:
        {
            "bin<bin_id>": { 
                "range": [start, end], 
                "#users": {
                    coldness<#interactions>": #users,
                    ...
                }
            }, 
            ...
        ]


    """
    # Define bin edges for the main interaction range
    num_edges = interaction_bins + 1
    bin_edges = np.linspace(min_interactions, max_interactions, num_edges)
    bin_edges = np.round(bin_edges).astype(int)
    
    # a bin for users with interactions higher than `max_interactions`
    total_bins = interaction_bins + 1

    # Initialize results dict
    results = {}
    for bin_id in range(total_bins):
        bin_key = f"bin{bin_id}"
        results[bin_key] = {}
        for coldness in simulated_coldness_levels:
            coldness_key = f"coldness{coldness}"
            results[bin_key][coldness_key] = {
                "users": [],
                "input_matrix": [],
                "input_items": [],
                "ground_truth_items": [],
            }
    
    actual_max_interactions = 0
    # Create masked rows for each user
    for user_id, pos_set in enumerate(test_positive_set):
        num_interactions: int = len(pos_set)

        if num_interactions < min_interactions:
            # In strong generalization, users not in the test set
            # are represented by rows of zeroes, we skip those here.
            continue

        if num_interactions > actual_max_interactions:
            actual_max_interactions = num_interactions
        
        # Determine bin for the user
        if num_interactions > max_interactions:
            bin_id = interaction_bins 
        else:
            # lower edge - exclusive, upper edge - inclusive
            bin_id = np.digitize(num_interactions, bin_edges, right=True) - 1
            if num_interactions == min_interactions:
                bin_id = 0
        bin_key = f"bin{bin_id}"
        
        pos_list = sorted(pos_set)
        for coldness in simulated_coldness_levels:
            if num_interactions > coldness:
                input_items = pos_list[:coldness]
                held_out_items = pos_list[coldness:]

                input_row = test_matrix[user_id, :].copy()
                input_row[0, held_out_items] = 0
                
                coldness_key = f"coldness{coldness}"
                results[bin_key][coldness_key]["users"].append(user_id)
                results[bin_key][coldness_key]["input_matrix"].append(input_row)
                results[bin_key][coldness_key]["input_items"].append(input_items)
                results[bin_key][coldness_key]["ground_truth_items"].append(set(held_out_items))
        
    # Make input items a sparse matrix 
    num_items = test_matrix.shape[1]
    for bin_key in results:
        for coldness_key in results[bin_key]:
            list_of_rows = results[bin_key][coldness_key]["input_matrix"]
            if list_of_rows:
                stacked_matrix = vstack(list_of_rows, format='csr')
            else:
                stacked_matrix = csr_matrix((0, num_items)) 
            results[bin_key][coldness_key]["input_matrix"] = stacked_matrix

    # Collect bin stats
    bin_stats = {} 
    for bin_id in range(total_bins):
        bin_key = f"bin{bin_id}"
        if bin_id < interaction_bins:
            start_range = bin_edges[bin_id]
            end_range = bin_edges[bin_id + 1]
        else:
            start_range = max_interactions
            end_range = actual_max_interactions

        coldness_stats = {}
        for coldness_key in results[bin_key]:
            coldness_stats[coldness_key] = len(results[bin_key][coldness_key]["users"])

        bin_stats[bin_key] = {
            "range": [start_range, end_range],
            "#users": coldness_stats
        }
        
    return results, bin_stats

def get_recommendations(logits: jax.Array, input_items: list[list[int]], k):
    # Set logits of input items to -inf.
    MINUS_INF = -float('inf')
    logits_masked = logits.copy()
    for i, item_ids in enumerate(input_items):
        logits_masked = logits_masked.at[i, item_ids].set(MINUS_INF)

    # Exclude -INF entries and select top-k
    recommended_item_indices = []
    for user_idx, user_logits in enumerate(logits):
        valid_indices = np.where(user_logits != MINUS_INF)[0]
        valid_logits = user_logits[valid_indices]
        
        # Get indices of top-k from valid logits
        top_indices = valid_indices[np.argsort(-valid_logits)[:k]]
        recommended_item_indices.append(top_indices)

    return recommended_item_indices