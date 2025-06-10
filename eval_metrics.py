import numpy as np
from sklearn.metrics import roc_auc_score

def hr(recommended_ranked_list: list[int], ground_truth_items: set[int], k: int) -> float:
    top_k_items = set(recommended_ranked_list[:k])
    hits_at_k = len(top_k_items & ground_truth_items)
    
    return float(hits_at_k) / len(ground_truth_items)

def ndcg(recommended_ranked_list: list[int], ground_truth_items: set[int], k: int) -> float:
    dcg, idcg = 0.0, 0.0

    for rank, item_idx in enumerate(recommended_ranked_list[:k]):
        if item_idx in ground_truth_items:
            dcg += 1.0 / np.log2(rank + 2)

    for rank in range(min(k, len(ground_truth_items))):
        idcg += 1.0 / np.log2(rank + 2)

    return dcg / idcg

def auc(y_true: list[int], y_score: list[float]) -> float:
    if len(set(y_true)) < 2:
        raise ValueError("ROC AUC is undefined with only one class in y_true.")
    return roc_auc_score(y_true, y_score)

def psp(
    recommended_ranked_list: list[int],
    ground_truth_items: set[int],
    item_propensities: list[float],
    k: int
) -> float:
    upsp, mpsp = 0.0, 0.0

    for item_idx in recommended_ranked_list[:k]:
        if item_idx in ground_truth_items:
            upsp += 1.0 / item_propensities[item_idx]
    upsp /= k

    for item_id in ground_truth_items:
        mpsp += 1.0 / item_propensities[item_id]

    return upsp / mpsp