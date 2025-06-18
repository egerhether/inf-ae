import numpy as np
from sklearn.metrics import roc_auc_score

"""
Original Paper Title: "Infinite Recommendation Networks: A Data-Centric Approach"
Main Text and Appendix: https://proceedings.neurips.cc/paper_files/paper/2022/hash/cac9e747a1d480c78312226959566cef-Abstract-Conference.html 
"""

def precision(recommended_ranked_list: list[int], ground_truth_items: set[int], k: int) -> float:
    top_k_items = set(recommended_ranked_list[:k])
    hits_at_k = len(top_k_items & ground_truth_items)
    
    if len(recommended_ranked_list) < k and len(recommended_ranked_list) != 0:
        k = len(recommended_ranked_list)
    
    return float(hits_at_k) / k

def recall(recommended_ranked_list: list[int], ground_truth_items: set[int], k: int) -> float:
    """
    What is defined in the paper. (Appendix B.5: "HitRate")
    """
    top_k_items = set(recommended_ranked_list[:k])
    hits_at_k = len(top_k_items & ground_truth_items)
    
    return float(hits_at_k) / len(ground_truth_items)

def truncated_recall(recommended_ranked_list: list[int], ground_truth_items: set[int], k: int) -> float:
    """
    What is actually computed. (eval.py line 74)
    """
    top_k_items = set(recommended_ranked_list[:k])
    hits_at_k = len(top_k_items & ground_truth_items)

    return float(hits_at_k) / min(k, len(ground_truth_items))

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
    """
    What is defined in the paper. (Appendix B.5: "Propensity-scored Precision")
    """
    upsp, mpsp = 0.0, 0.0
    denum = float(min(k, len(recommended_ranked_list)))

    for item_idx in recommended_ranked_list[:k]:
        if item_idx in ground_truth_items:
            upsp += (1.0 / item_propensities[item_idx]) / denum

    for item_id in ground_truth_items:
        mpsp += 1.0 / item_propensities[item_id]

    return upsp / mpsp

def capped_psp(
    recommended_ranked_list: list[int],
    ground_truth_items: set[int],
    item_propensities: list[float],
    k: int
) -> float:
    """
    What actually computed. (eval.py lines 76- 88)
    """
    upsp, max_psp = 0.0, 0.0
    denum = float(min(k, len(ground_truth_items))) # difference 1: normalization of upsp

    ground_truth_inv_propensity_sorted = sorted([ 1.0 / item_propensities[x] for x in ground_truth_items ])[::-1]

    for at, item_idx in enumerate(recommended_ranked_list[:k]):
        if item_idx in ground_truth_items: 
            upsp += (1.0 / item_propensities[item_idx]) / denum
        if at < len(ground_truth_items):
            max_psp += ground_truth_inv_propensity_sorted[at] # difference 2: upsp normalized by 

    return upsp / max_psp

class GiniCoefficient:
    """
    A class to calculate the Gini coefficient, a measure of income inequality.
    The Gini coefficient ranges from 0 (perfect equality) to 1 (perfect inequality).
    """

    def gini_coefficient(self, values):
        """
        Compute the Gini coefficient of array of values.
        For a frequency vector, G = sum_i sum_j |x_i - x_j| / (2 * n^2 * mu)
        """
        # print(f"[GINI] Computing Gini coefficient for {len(values)} values")
        arr = np.array(values, dtype=float)
        if arr.sum() == 0:
            # print("[GINI] Sum of values is 0, returning 0.0")
            return 0.0
        # sort and normalize
        arr = np.sort(arr)
        n = arr.size
        cumvals = np.cumsum(arr)
        mu = arr.mean()
        # the formula simplifies to:
        # G = (1 / (n * mu)) * ( sum_i (2*i - n - 1) * arr[i] )
        index = np.arange(1, n + 1)
        gini = (np.sum((2 * index - n - 1) * arr)) / (n * n * mu)
        # print(f"[GINI] Computed Gini coefficient: {gini:.4f}")
        return gini

    def calculate_list_gini(self, articles, key="category"):
        """
        Given a list of article dicts and a key (e.g. 'category'), compute the
        Gini coefficient over the frequency distribution of that key.
        """
        # print(f"[GINI] Calculating Gini for {len(articles)} articles using key '{key}'")s
        # count frequencies
        freqs = {}
        for art in articles:
            val = art.get(key, None) or "UNKNOWN"
            freqs[val] = freqs.get(val, 0) + 1
        # print(f"[GINI] Found {len(freqs)} unique {key} values")
        return self.gini_coefficient(list(freqs.values()))