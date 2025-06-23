import numpy as np
from scipy.sparse import vstack, csr_matrix
from collections import defaultdict
from jax import Array as jaxArray
import json
import os

import eval_metrics
from data import Dataset
from utils import get_cores

MINUS_INF = -float('inf')

def run_cold_start_experiment(
        data: Dataset,
        hyper_params,
        krr_forward,
        train_matrix,
        precomputed_alpha,
        k_values = [10, 100]
    ):

    dataset_cores = get_cores(
            data.data["train_positive_set"],
            data.data["val_positive_set"],
            data.data["test_positive_set"]
        )

    most_popular_train_items = get_popular_items(
        data.data["train_matrix"],
        dataset_cores + max(hyper_params["simulated_coldness_levels"])
    )

    popular_diverse_train_items = get_popular_diverse_items(
        data.data["train_matrix"],
        data.data["item_map_to_category"],
        dataset_cores + max(hyper_params["simulated_coldness_levels"])
    )

    cold_start_splits, cold_start_stats = prepare_cold_start_data(
        data.data["test_positive_set"],
        data.data["test_matrix"],
        dataset_cores,
        hyper_params["simulated_max_interactins"],
        hyper_params["cold_start_bins"],
        hyper_params["simulated_coldness_levels"],
        most_popular_train_items,
        popular_diverse_train_items,
    )

    baseline_metrics = evaluate(
        k_values,
        cold_start_splits,
        cold_start_stats,
        krr_forward,
        train_matrix,
        precomputed_alpha,
        data,
        hyper_params,
        "input_matrix"
    )

    popular_item_filled_metrics = evaluate(
        k_values,
        cold_start_splits,
        cold_start_stats,
        krr_forward,
        train_matrix,
        precomputed_alpha,
        data,
        hyper_params,
        "popular_input_matrix"
    )

    popular_diverse_item_filled_metrics = evaluate(
        k_values,
        cold_start_splits,
        cold_start_stats,
        krr_forward,
        train_matrix,
        precomputed_alpha,
        data,
        hyper_params,
        "popular_diverse_input_matrix"
    )

    dir = f"./results/cold-start/{hyper_params['dataset']}/seed{hyper_params['seed']}/"
    os.makedirs(dir, exist_ok=True)
    with open(f"{dir}stats.json", "w") as f:
        f.write(json.dumps(cold_start_stats, indent=4))
    with open(f"{dir}baseline_metrics.json", "w") as f:
        f.write(json.dumps(baseline_metrics, indent=4))
    with open(f"{dir}popular_filled_metrics.json", "w") as f:
        f.write(json.dumps(popular_item_filled_metrics, indent=4))
    with open(f"{dir}popular_diverse_filled_metrics.json", "w") as f:
        f.write(json.dumps(popular_diverse_item_filled_metrics, indent=4))
    
def get_popular_items(train_matrix: csr_matrix, k: int) -> list[int]:
    """
    Gets the top-k most popular item ids based on the train matrix.

    Args:
        train_matrix: A binary matrix representing user interactions from the train set.
            Each row is a user, each column is an item.
        k: The number of top popular items to return.

    Returns:
        A list of the integer IDs of the top-k most popular items, sorted from
        most to least popular.
    """
    item_popularity_matrix = train_matrix.sum(axis=0)
    item_popularity = item_popularity_matrix.A1 # flatten
    top_k_item_indices = np.argsort(item_popularity)[-k:][::-1]
    return top_k_item_indices.tolist()

def get_popular_diverse_items(
    train_matrix: csr_matrix,
    item_tag_mapping: dict[int, str],
    k: int
) -> list[int]:
    """
    Gets the top-k items by selecting one item at a time from categories
    in a round-robin fashion, ordered by category popularity.

    The algorithm works as follows:
    1.  Calculates the popularity of each item.
    2.  Groups items by category and sorts them by popularity within each category.
    3.  Calculates the total popularity of each category.
    4.  Orders the categories from most to least popular.
    5.  Iterates through the sorted categories, picking the top available item from
        each one, until k items have been selected. This ensures the most popular
        categories get their top items included first, while still giving less
        popular categories a chance to be represented.

    Args:
        train_matrix: A binary matrix of user-item interactions.
        item_tag_mapping: A dictionary mapping item_id (int) to its category/tag_id (string but
    such that the distribution of item categories is as uniform as possible diverse).
        k: The number of items to return.

    Returns:
        A list of the integer IDs of the top-k most popular diverse items,
        ordered by the round-robin selection process.
    """
    item_popularity = train_matrix.sum(axis=0).A1
    
    category_to_items = defaultdict(list)
    for item_id, pop in enumerate(item_popularity):
        category_id = item_tag_mapping.get(item_id)
        if category_id is not None:
            category_to_items[category_id].append((pop, item_id))
    
    for category_id in category_to_items:
        category_to_items[category_id].sort(key=lambda x: x[0], reverse=True)

    category_popularity = {
        cat_id: sum(pop for pop, item in items)
        for cat_id, items in category_to_items.items()
    }

    sorted_categories = sorted(
        category_popularity.keys(),
        key=lambda cat_id: category_popularity[cat_id],
        reverse=True
    )

    top_k_item_indices = []
    selected_item_ids = set()
    category_item_pointer = defaultdict(int) 

    while len(top_k_item_indices) < k:
        items_added_in_this_round = 0
        for category_id in sorted_categories:
            items_in_category = category_to_items[category_id]
            pointer = category_item_pointer[category_id]
            
            if pointer < len(items_in_category):
                _, item_id = items_in_category[pointer]
                
                if item_id not in selected_item_ids:
                    top_k_item_indices.append(item_id)
                    selected_item_ids.add(item_id)
                    items_added_in_this_round += 1

                category_item_pointer[category_id] += 1

                if len(top_k_item_indices) == k:
                    break
        
        if items_added_in_this_round == 0:
            break

    return top_k_item_indices

def stack_sparse_rows_into_matrix(list_of_rows: list[csr_matrix], num_items):
    if list_of_rows:
        return vstack(list_of_rows, format='csr')
    return csr_matrix((0, num_items))

def fill_input_row_with(
    row_to_fill: csr_matrix,
    input_items: list[int],
    filler_items: list[int],
    min_interactions: int,
    coldness_level: int
):
    valid_filler_items = [fi for fi in filler_items if fi not in input_items]
    fill_with = valid_filler_items[:min_interactions - coldness_level]
    row_to_fill[0, fill_with] = 1
    assert (len(fill_with) + len(input_items)) == min_interactions

def prepare_cold_start_data(
    test_positive_set: list[set[int]],
    test_matrix: csr_matrix,
    min_interactions: int,
    max_interactions: int,
    interaction_bins: int,
    simulated_coldness_levels: list[int],
    popular_items: list[int],
    popular_diverse_items: list[int],
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
        max_interactions: Upper bound for the last interaction bin.
        interaction_bins: Number of bins to create between min and max interactions.
        simulated_coldness_levels: A list of integers, where each integer represents 
            a number of interactions to be used as input for the model, simulating a
            user with that level of "coldness".
        popular_items: A list of popular item ids to be used to filling in sparse user 
            input vectors until they reach `min_interactions` number of interactions.
        popular_diverse_items: A list of popular diverse item ids to be used to filling in
            sparse user input vectors until they reach `min_interactions` number of interactions.
    Returns:
        A dictionary structured for evaluating cold-start performance. The structure is:
        {
            "bin<bin_id>": {
                "coldness<#interactions>": {
                    "users": [user_id_1, user_id_2, ...],
                    "input_matrix": <scipy.sparse.matrix of shape [#users, #items]>,
                    "input_items": [[items_ids...], [items_ids...], ...],
                    "ground_truth_items": [[items_ids...], [items_ids...], ...],
                    "popular_input_matrix": <scipy.sparse.matrix of shape [#users, #items]>,
                    "popular_diverse_input_matrix": <scipy.sparse.matrix of shape [#users, #items]>,  
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

    # Initialize results dict
    results = {}
    for bin_id in range(interaction_bins):
        bin_key = f"bin{bin_id}"
        results[bin_key] = {}
        for coldness in simulated_coldness_levels:
            coldness_key = f"coldness{coldness}"
            results[bin_key][coldness_key] = {
                "users": [],
                "input_matrix": [],
                "input_items": [],
                "ground_truth_items": [],
                "popular_input_matrix": [],
                "popular_diverse_input_matrix": [],
            }
    
    actual_max_interactions = 0
    # Create masked rows for each user
    for user_id, pos_set in enumerate(test_positive_set):
        num_interactions: int = len(pos_set)

        if num_interactions < min_interactions:
            # In strong generalization, users not in the test set
            # are represented by rows of zeroes, we skip those here.
            continue

        if num_interactions > max_interactions:
            continue

        if num_interactions > actual_max_interactions:
            actual_max_interactions = num_interactions
        
        # Determine bin for the user
        # lower edge - exclusive, upper edge - inclusive
        bin_id = np.digitize(num_interactions, bin_edges, right=True) - 1
        if num_interactions == min_interactions:
            bin_id = 0
        bin_key = f"bin{bin_id}"
        
        pos_list = sorted(pos_set)
        for coldness in simulated_coldness_levels:
            if num_interactions > coldness:
                # Baseline cold-start data
                input_items = pos_list[:coldness]
                held_out_items = pos_list[coldness:]
                input_row = test_matrix[user_id, :].copy()
                input_row[0, held_out_items] = 0
                
                coldness_key = f"coldness{coldness}"
                results[bin_key][coldness_key]["users"].append(user_id)
                results[bin_key][coldness_key]["input_matrix"].append(input_row)
                results[bin_key][coldness_key]["input_items"].append(input_items)
                results[bin_key][coldness_key]["ground_truth_items"].append(set(held_out_items))

                popular_item_filled_row = input_row.copy()
                fill_input_row_with(popular_item_filled_row, input_items, popular_items, min_interactions, coldness)
                results[bin_key][coldness_key]["popular_input_matrix"].append(popular_item_filled_row)

                popular_diverse_filled_row = input_row.copy()
                fill_input_row_with(popular_diverse_filled_row, input_items, popular_diverse_items, min_interactions, coldness)
                results[bin_key][coldness_key]["popular_diverse_input_matrix"].append(popular_diverse_filled_row)

    # Make input items a sparse matrix 
    num_items = test_matrix.shape[1]
    for bin_key in results:
        for coldness_key in results[bin_key]:
            for matrix_key in ["input_matrix", "popular_input_matrix", "popular_diverse_input_matrix"]:
                results[bin_key][coldness_key][matrix_key] = stack_sparse_rows_into_matrix(
                    results[bin_key][coldness_key][matrix_key], num_items)

    # Collect bin stats
    bin_stats = {} 
    for bin_id in range(interaction_bins):
        bin_key = f"bin{bin_id}"
        start_range = bin_edges[bin_id]
        end_range = bin_edges[bin_id + 1]

        coldness_stats = {}
        for coldness_key in results[bin_key]:
            coldness_stats[coldness_key] = len(results[bin_key][coldness_key]["users"])

        bin_stats[bin_key] = {
            "range": [int(start_range), int(end_range)],
            "#users": coldness_stats
        }
        
    return results, bin_stats

def get_recommendations(logits: jaxArray, input_items: list[list[int]], k: int) -> list[list[int]]:
    """
    Generates top-k recommendations, excluding items the user has already interacted with.

    Args:
        logits: A JAX array of shape (#users, #items) with the model's prediction scores.
        input_items: A list of lists, where each inner list contains the item IDs used as input for a user.
        k: The number of recommendations to generate.

    Returns:
       A list of lists of integers. Each inner list contains up to `k`
       recommended item IDs for a user, sorted by score. The lists can be
       shorter than `k` if fewer than `k` items are available to recommend. 
    """
    num_users, num_items= logits.shape
    if k == 0:
        raise ValueError(f"k = 0 is not allowed.")

    # Set the score of input items to -inf for each user.
    logits_masked = np.array(logits) 
    for i, item_ids in enumerate(input_items):
        if item_ids:
            logits_masked[i, item_ids] = MINUS_INF

    # Partition to get the top k items (will be at the end of the array)
    candidate_indices = np.argpartition(logits_masked, -k, axis=1)[:, -k:]
    row_indices_for_gather = np.arange(num_users)[:, np.newaxis]
    candidate_scores = logits_masked[row_indices_for_gather, candidate_indices]

    # Filter invalid items and sort
    final_recommendations = []
    for u in range(num_users):
        user_indices = candidate_indices[u]
        user_scores = candidate_scores[u]

        # Filter to get only valid indices and their corresponding scores
        is_valid = user_scores > MINUS_INF
        valid_indices = user_indices[is_valid]
        valid_scores = user_scores[is_valid]
        
        # Sort the valid items by score in descending order
        final_order = np.argsort(-valid_scores)
        ranked_recs = valid_indices[final_order]
        
        final_recommendations.append(ranked_recs.tolist())
        
    return final_recommendations

def evaluate(
        k_values: list[int],
        cold_start_splits: dict,
        cold_start_stats: dict,
        krr_forward,
        train_matrix: csr_matrix,
        precomputed_alpha,
        data: Dataset,
        hyper_params: dict,
        input_matrix_key
        ):
    max_k = max(k_values)
    metrics = {}
    for (bin_key, coldness_splits) in cold_start_splits.items():
        metrics[bin_key] = {}
        for (coldness_key, split) in coldness_splits.items():
            metrics[bin_key][coldness_key] = {}

            logits = krr_forward(
                X_train=train_matrix,
                X_predict=split[input_matrix_key].todense(),
                reg=hyper_params["lamda"],
                alpha=precomputed_alpha,
            )

            total_metrics =  defaultdict(float)
            global_auc_labels, global_auc_scores = [], []
            global_category_counts = {}
            for k in k_values:
                global_category_counts[k] = {}

            recommendations = get_recommendations(logits, split["input_items"], max_k)
            for u in range(len(recommendations)):
                orig_user_id = split["users"][u]
                auc_result, auc_labels, auc_scores = eval_metrics.auc_with_prep(np.array(logits[u]), split["ground_truth_items"][u], data.data["negatives"][orig_user_id])
                total_metrics["MEAN_AUC"] += auc_result
                global_auc_labels.extend(auc_labels)
                global_auc_scores.extend(auc_scores)

                for k in k_values:
                    total_metrics[f"PRECISION@{k}"] += eval_metrics.precision(recommendations[u], split["ground_truth_items"][u], k)
                    total_metrics[f"RECALL@{k}"] += eval_metrics.recall(recommendations[u], split["ground_truth_items"][u], k)
                    total_metrics[f"TRUNCATED_RECALL@{k}"] += eval_metrics.truncated_recall(recommendations[u], split["ground_truth_items"][u], k)
                    total_metrics[f"NDCG@{k}"] += eval_metrics.ndcg(recommendations[u], split["ground_truth_items"][u], k)
                    if "item_tag_mapping" in data.data and len(data.data["item_tag_mapping"]) > 0:
                        category_recommendations = eval_metrics.prepare_category_counts(recommendations[u], data.data["item_tag_mapping"], k)
                        total_metrics[f"ILD@{k}"] += eval_metrics.intra_list_jaccard_distance(recommendations[u], data.data["item_tag_mapping"], k)
                        total_metrics[f"ENTROPY@{k}"] += eval_metrics.entropy(list(category_recommendations.values()))
                        total_metrics[f"GINI@{k}"] += eval_metrics.gini(list(category_recommendations.values()))

                        for category, count in category_recommendations.items():
                            global_category_counts[k][category] = global_category_counts[k].get(category, 0) + count
                
            metrics[bin_key][coldness_key]["GLOBAL_AUC"] = eval_metrics.auc(global_auc_labels, global_auc_scores)
            for k in k_values:
                category_counts_array = list(global_category_counts[k].values())
                metrics[bin_key][coldness_key][f"GLOBAL_GINI@{k}"] = eval_metrics.gini(category_counts_array)
                metrics[bin_key][coldness_key][f"GLOBAL_ENTROPY@{k}"] = eval_metrics.entropy(category_counts_array)

            num_users = cold_start_stats[bin_key]["#users"][coldness_key]
            for metric in total_metrics:
                metrics[bin_key][coldness_key][metric] = total_metrics[metric] / num_users

    return metrics 