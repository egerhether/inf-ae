import numpy as np
import warnings
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

def inter_list_jaccard_distance(
    recommended_ranked_list: list[int], 
    item_tag_mapping: dict, 
    k: int
) -> float:
    """
    Calculate Inter-list distance using Jaccard distance based on item tags.
    
    Args:
        recommended_ranked_list: List of recommended item IDs ranked by score
        item_tag_mapping: Dictionary mapping item_id -> set of tags (any hashable type)
        k: Number of top recommendations to consider
        
    Returns:
        Average Jaccard distance between all pairs of items in top-k recommendations
    """

    if k == 0:
        result = 0.0
    elif len(recommended_ranked_list) == 0:
        warnings.warn("Empty recommendation list provided", UserWarning)
        result = 0.0
    else:
        top_k_items = recommended_ranked_list[:k]
        
        if len(top_k_items) < 2:
            result = 0.0  # Cannot compute distance with fewer than 2 items
        else:
            distances = []
            for i in range(len(top_k_items)):
                for j in range(i + 1, len(top_k_items)):
                    item_i, item_j = top_k_items[i], top_k_items[j]
                    
                    # Get tags, defaulting to empty set if item not in mapping
                    tags_i = set(item_tag_mapping.get(item_i, set()))
                    tags_j = set(item_tag_mapping.get(item_j, set()))
                    
                    # Jaccard distance = 1 - Jaccard similarity
                    intersection = len(tags_i & tags_j)
                    union = len(tags_i | tags_j)
                    
                    if union == 0:  # Both items have no tags
                        jaccard_distance = 0.0
                    else:
                        jaccard_similarity = intersection / union
                        jaccard_distance = 1.0 - jaccard_similarity
                    
                    distances.append(jaccard_distance)
            
            result = sum(distances) / len(distances) if distances else 0.0
    
    return result

def prepare_category_counts(
    recommended_ranked_list: list[int], 
    item_tag_mapping: dict, 
    k: int
) -> list[int]:
    """
    Prepare category counts from recommended items for entropy calculation.
    
    Args:
        recommended_ranked_list: List of recommended item IDs ranked by score
        item_tag_mapping: Dictionary mapping item_id -> category/tag (or set of categories)
        k: Number of top recommendations to consider
        
    Returns:
        List of counts for each category found in the recommendations
    """
    if k == 0 or len(recommended_ranked_list) == 0:
        return []
    
    top_k_items = recommended_ranked_list[:k]
    category_counts = {}
    
    for item_id in top_k_items:
        if item_id in item_tag_mapping:
            # Handle both single category and multiple categories (sets)
            categories = item_tag_mapping[item_id]
            if isinstance(categories, (set, list, tuple)) and not isinstance(categories, str):
                # weight = 1.0 / len(categories)  # Fractional weight
                for category in categories:
                    category_counts[category] = category_counts.get(category, 0) + 1
            else:
                # Single category per item
                category_counts[categories] = category_counts.get(categories, 0) + 1
        else:
            # Item not in mapping, assign to "UNKNOWN" category
            category_counts["UNKNOWN"] = category_counts.get("UNKNOWN", 0) + 1

    return list(category_counts.values())

def entropy(category_counts: list) -> float:
    """
    Calculate the entropy of game category recommendations to measure how evenly 
    recommendations are distributed across categories.
    
    Args:
        category_counts: List of counts for each category
        
    Returns:
        Shannon entropy value (bits). Higher values indicate more diverse and 
        balanced recommendations, while lower values suggest bias toward few categories.
        
    Raises:
        ValueError: If category_counts contains negative values or all counts are zero
        ZeroDivisionError: If category_counts is empty
    """
    if not category_counts:
        raise ValueError("Category counts cannot be empty")
    
    # Convert to numpy array for easier handling
    counts = np.array(category_counts, dtype=float)
    
    # Check for negative values
    if np.any(counts < 0):
        raise ValueError("Category counts cannot be negative")
    
    # Remove zero counts (they don't contribute to entropy)
    nonzero_counts = counts[counts > 0]
    
    if len(nonzero_counts) == 0:
        raise ValueError("All category counts are zero")
    
    # Calculate total
    total = np.sum(nonzero_counts)
    
    if total == 0:
        raise ValueError("Total count cannot be zero")
    
    # Calculate probabilities
    probabilities = nonzero_counts / total
    
    # Calculate Shannon entropy using log base 2
    entropy_value = -np.sum(probabilities * np.log2(probabilities))

    # normalized_entropy = entropy_value / np.log2(len(nonzero_counts)) if len(nonzero_counts) > 1 else 0.0

    return float(entropy_value)

def gini(category_counts: list) -> float:
    
    values = np.sort(category_counts)
    n = len(values)

    # gini compute
    cumulative_sum = np.cumsum(values)
    gini = (n + 1 - 2 * (np.sum(cumulative_sum) / cumulative_sum[-1])) / n

    return gini