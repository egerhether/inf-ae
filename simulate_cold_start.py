import numpy as np

def cold_start_binning(
    test_positive_set: list[set[int]],
    min_interactions: int,
    max_interactions: int,
    interaction_bins: int,
    simulated_coldness_levels: list[int],
):
    """
    - Splits users into bins based on their number of interactions.
    - Bins are of equal length and all belong to the [min_interactions, max_interactions] interval.    
    - Each user history is used to simulate the cold-start/low-interaction-level scenario where a
    small part of the item history is reserved to be used as input to Inf-AE and the rest is
    reserved for evaluation.

    Assumptions: 
        1. Strong generalization evaluation setup. <-> The positive set is the full interaction history.
        2. `min_interactions` and `max_interactions` are global values based on the full dataset.

    Args:
        test_positive_set: List of sets of item ids representing user-items histories.
        min_interactions: Lower bound for the first interaction bin. Users are assumed to not be under this.
        max_interactions: Upper bound for the last interaction bin. All users exceeding this are added to the last bin.
        interaction_bins: Number bins to create in [`min_interactions`, `max_interactions`].
        simulated_coldness_levels: A list of integers, where each integer 
            represents a number of interactions to be used as input for the 
            model, simulating a user with that level of "coldness".

    Returns:
        A dictionary structured for evaluating cold-start performance. The structure is:
        {
          "bin<bin_id>": {
              "coldness<#interactions>": {
                  "users": [user_id_1, user_id_2, ...],
                  "input_items": [[item_ids...], [items_ids...], ...],
                  "predict_items": [[items_ids...], [items_ids...], ...]
              }, ...
          }, ...
          "bin_stats": {
              0: {
                "range": [min, max],
                "#users": <#users>,
              }
              ...
          }
        }
    """
    # Calculate bin edges based on max_interactions 
    bin_edges = np.linspace(min_interactions, max_interactions, interaction_bins + 1)
    bin_edges = np.round(bin_edges).astype(int) 

    # Initialize results dict
    results = {
        "bin_stats": {}
    }
    for i in range(interaction_bins):
        bin_key = f"bin{i}"
        results[bin_key] = {}

        for coldness in simulated_coldness_levels:
            coldness_key = f"coldness{coldness}"
            results[bin_key][coldness_key] = {
                "users": [],
                "input_items": [],
                "predict_items": []
            }
    
    skipped = 0

    for user_id, pos_set in enumerate(test_positive_set):
        num_interactions: int = len(pos_set)
        if num_interactions < min_interactions:
            skipped = skipped + 1
            continue

        if num_interactions > bin_edges[-1]:
            bin_edges[-1] = num_interactions

        # Find the bin to which the user belongs to.
        # Subtract one as `np.digitize` is 1-based.
        bin_id = np.digitize(num_interactions, bin_edges, right=True) - 1
        if num_interactions == min_interactions:
            bin_id = 0
        bin_key = f"bin{bin_id}"

        # Sort for deterministic slicing
        pos_list = sorted(pos_set)

        # Create cold-start scenarios.
        for coldness in simulated_coldness_levels:
            # User must have more interactions than the coldness level
            # to have items left to predict.
            if num_interactions > coldness:
                coldness_key = f"coldness{coldness}"
                
                input_items = pos_list[:coldness]
                items_to_predict = pos_list[coldness:]

                results[bin_key][coldness_key]["users"].append(user_id)
                results[bin_key][coldness_key]["input_items"].append(input_items)
                results[bin_key][coldness_key]["predict_items"].append(items_to_predict)
    
    for i in range(interaction_bins):
        start_range = bin_edges[i]
        end_range = bin_edges[i+1]

        results["bin_stats"][i] = {
            "range": [start_range, end_range],
            "#users": len(results[f"bin{i}"][f"coldness{simulated_coldness_levels[0]}"]["users"])
        }
        
    # print(
    #     f"[COLD] {skipped} users were skipped because they had less interactions than "
    #     f"{min_interactions} interactions. Should not happen for weak generazlization."
    # )
    return results