def calculate_hit_rate(recommended_ranked_list: list[int], ground_truth_items: set[int], k: int):
    top_k_items = set(recommended_ranked_list[:k])
    hits_at_k = len(top_k_items & ground_truth_items)
    return float(hits_at_k) / float(len(ground_truth_items))