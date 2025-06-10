import numpy as np

def calculate_hit_rate(recommended_ranked_list: list[int], ground_truth_items: set[int], k: int):
    top_k_items = set(recommended_ranked_list[:k])
    hits_at_k = len(top_k_items & ground_truth_items)
    
    return float(hits_at_k) / float(len(ground_truth_items))

def calculate_ndcg(recommended_ranked_list: list[int], ground_truth_items: set[int], k: int):
    dcg, idcg = 0.0, 0.0

    for rank, item_idx in enumerate(recommended_ranked_list[:k]):
        if item_idx in ground_truth_items:
            dcg += 1.0 / np.log2(rank + 2)

    for rank in range(min(k, len(ground_truth_items))):
        idcg += 1.0 / np.log2(rank + 2)

    return dcg / idcg