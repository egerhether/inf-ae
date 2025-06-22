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

def intra_list_jaccard_distance(
    recommended_ranked_list: list[int], 
    item_tag_mapping: dict, 
    k: int
) -> float:
    """
    Calculate Intra-list distance (ILD) using Jaccard distance based on item tags.

    Args:
        recommended_ranked_list: List of recommended item IDs ranked by score
        item_tag_mapping: Dictionary mapping item_id -> set of tags (any hashable type)
        k: Number of top recommendations to consider
        
    Returns:
        Average Jaccard distance between all pairs of items in top-k recommendations
    """

    if k == 0:
        raise ValueError("k must be greater than 0")
    elif len(recommended_ranked_list) == 0:
        raise ValueError("recommended_ranked_list cannot be empty")
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
            
            result = sum(distances) / len(distances)
    
    return result

def prepare_category_counts(
    recommended_ranked_list: list[int], 
    item_tag_mapping: dict, 
    k: int
) -> list[int]:
    """
    Prepare category counts from recommended items for entropy and gini calculation.
    
    Args:
        recommended_ranked_list: List of recommended item IDs ranked by score
        item_tag_mapping: Dictionary mapping item_id -> category/tag (or set of categories)
        k: Number of top recommendations to consider
        
    Returns:
        Dict of counts for each category found in the recommendations
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
                # weight = 1.0 / len(categories)  # Fractional weight optional, if needed
                for category in categories:
                    category_counts[category] = category_counts.get(category, 0) + 1
            else:
                # Single category per item
                category_counts[categories] = category_counts.get(categories, 0) + 1
        else:
            # Item not in mapping, assign to "UNKNOWN" category
            category_counts["UNKNOWN"] = category_counts.get("UNKNOWN", 0) + 1

    return category_counts

def entropy(category_counts: list) -> float:
    """
    Calculate the entropy of category recommendations to measure how evenly 
    recommendations are distributed across categories.
    
    Args:
        category_counts: List of counts for each category
        
    Returns:
        Shannon entropy value (bits). Higher values indicate more diverse and 
        balanced recommendations, while lower values suggest bias toward few categories.
        
    Raises:
        ValueError: If category_counts contains negative values or all counts are zero
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
    
    # Calculate probabilities
    probabilities = nonzero_counts / total
    
    # Calculate Shannon entropy using log base 2
    entropy_value = -np.sum(probabilities * np.log2(probabilities))

    normalized_entropy = entropy_value / np.log2(len(nonzero_counts)) if len(nonzero_counts) > 1 else 0.0

    return float(normalized_entropy)

def gini(category_counts: list) -> float:
    if not category_counts:
        raise ValueError("Category counts cannot be empty")
    
    counts = np.array(category_counts, dtype=float)

    # Check for negative values
    if np.any(counts < 0):
        raise ValueError("Category counts cannot be negative")
    
    # Remove zero counts (they don't contribute to Gini)
    nonzero_counts = counts[counts > 0]

    # Check for empty nonzero counts
    if len(nonzero_counts) == 0:
        raise ValueError("All category counts are zero")

    gini = 1 - np.sum((nonzero_counts/np.sum(nonzero_counts))**2)
    return gini