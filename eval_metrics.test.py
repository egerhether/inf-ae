import unittest
import numpy as np
import warnings

import eval_metrics

class TestCalculatePrecision(unittest.TestCase):
    def test_no_hits(self):
        recommendations = [101, 102, 103]
        ground_truth = {1, 2, 3}
        k = 3
        self.assertAlmostEqual(eval_metrics.precision(recommendations, ground_truth, k), 0.0)

    def test_all_hits(self):
        recommendations = [1, 2, 3]
        ground_truth = {1, 2, 3}
        k = 3
        self.assertAlmostEqual(eval_metrics.precision(recommendations, ground_truth, k), 1.0)

    def test_partial_hits(self):
        recommendations = [1, 2, 4]
        ground_truth = {1, 2, 3}
        k = 3
        self.assertAlmostEqual(eval_metrics.precision(recommendations, ground_truth, k), 2/3)

    def test_k_is_zero(self):
        recommendations = [1, 2, 3]
        ground_truth = {1, 2}
        k = 0
        with self.assertRaises(ZeroDivisionError):
            eval_metrics.precision(recommendations, ground_truth, k)

    def test_k_is_larger_than_recommendation_list(self):
        recommendations = [1, 2]
        ground_truth = {1, 3}
        k = 10
        self.assertAlmostEqual(eval_metrics.precision(recommendations, ground_truth, k), 0.5)

    def test_empty_recommendation_list(self):
        recommendations = []
        ground_truth = {1, 2}
        k = 5
        self.assertAlmostEqual(eval_metrics.precision(recommendations, ground_truth, k), 0.0)
    

class TestCalculateRecallAndTruncatedVersion(unittest.TestCase):

    def test_no_hits(self):
        recommendations = [101, 102, 103]
        ground_truth = {1, 2, 3}
        k = 3
        self.assertAlmostEqual(eval_metrics.recall(recommendations, ground_truth, k), 0.0)
        self.assertAlmostEqual(eval_metrics.truncated_recall(recommendations, ground_truth, k), 0.0)

    def truncate_denum(self):
        recommendations = [1, 2, 3, 4, 5]
        ground_truth = {3, 5, 10, 100, 500}
        k = 4
        self.assertAlmostEqual(eval_metrics.recall(recommendations, ground_truth, k), 1/5)
        self.assertAlmostEqual(eval_metrics.truncated_recall(recommendations, ground_truth, k), 1/4)

    def test_all_gt_in_k(self):
        recommendations = [10, 20, 30, 40, 50]
        ground_truth = {20, 40}
        k = 5
        self.assertAlmostEqual(eval_metrics.recall(recommendations, ground_truth, k), 1.0)
        self.assertAlmostEqual(eval_metrics.truncated_recall(recommendations, ground_truth, k), 1.0)

    def test_no_hits_in_k(self):
        recommendations = [1, 2, 3, 4, 5]
        ground_truth = {4, 5, 6}
        k = 2
        self.assertAlmostEqual(eval_metrics.recall(recommendations, ground_truth, k), 0.0)
        self.assertAlmostEqual(eval_metrics.truncated_recall(recommendations, ground_truth, k), 0.0)

    def test_k_is_zero(self):
        recommendations = [1, 2, 3]
        ground_truth = {1, 2}
        k = 0
        self.assertAlmostEqual(eval_metrics.recall(recommendations, ground_truth, k), 0.0)
        with self.assertRaises(ZeroDivisionError):
            eval_metrics.truncated_recall(recommendations, ground_truth, k)

    def test_k_is_larger_than_recommendation_list(self):
        recommendations = [1, 2]
        ground_truth = {1, 3}
        k = 10
        self.assertAlmostEqual(eval_metrics.recall(recommendations, ground_truth, k), 0.5)
        self.assertAlmostEqual(eval_metrics.truncated_recall(recommendations, ground_truth, k), 0.5)

    def test_empty_recommendation_list(self):
        recommendations = []
        ground_truth = {1, 2, 3}
        k = 5
        self.assertAlmostEqual(eval_metrics.recall(recommendations, ground_truth, k), 0.0)
        self.assertAlmostEqual(eval_metrics.truncated_recall(recommendations, ground_truth, k), 0.0)

    def test_empty_ground_truth_raises_error(self):
        recommendations = [1, 2, 3]
        ground_truth = set()
        k = 3
        with self.assertRaises(ZeroDivisionError):
            eval_metrics.recall(recommendations, ground_truth, k)
        with self.assertRaises(ZeroDivisionError):
            eval_metrics.truncated_recall(recommendations, ground_truth, k)

class Testeval_CalculateNDCG(unittest.TestCase):

    def test_perfect_ranking_is_one(self):
        recommendations = [1, 2, 3, 4, 5]
        ground_truth = {1, 2, 3}
        k = 3
        self.assertAlmostEqual(eval_metrics.ndcg(recommendations, ground_truth, k), 1.0)

    def test_no_hits_is_zero(self):
        recommendations = [101, 102, 103]
        ground_truth = {1, 2, 3}
        k = 3
        self.assertAlmostEqual(eval_metrics.ndcg(recommendations, ground_truth, k), 0.0)

    def test_typical_good_ranking(self):
        recommendations = [1, 99, 2, 98, 3] # Hits at pos 1, 3, 5
        ground_truth = {1, 2, 3}
        k = 5
        expected_dcg = 1 / np.log2(1 + 1) + 1 / np.log2(3 + 1) + 1 / np.log2(5 + 1)
        expected_idcg = 1 / np.log2(1 + 1) + 1 / np.log2(2 + 1) + 1 / np.log2(3 + 1)
        expected_ndcg = expected_dcg / expected_idcg
        self.assertAlmostEqual(eval_metrics.ndcg(recommendations, ground_truth, k), expected_ndcg)

    def test_k_is_smaller_than_number_of_hits(self):
        recommendations = [1, 2, 3, 4, 5]
        ground_truth = {1, 2, 3, 4, 5}
        k = 3
        self.assertAlmostEqual(eval_metrics.ndcg(recommendations, ground_truth, k), 1.0)
        
    def test_empty_recommendations_is_zero(self):
        recommendations = []
        ground_truth = {1, 2, 3}
        k = 5
        self.assertAlmostEqual(eval_metrics.ndcg(recommendations, ground_truth, k), 0.0)

    def test_k_is_zero_raises_error(self):
        recommendations = [1, 2, 3]
        ground_truth = {1, 2}
        k = 0
        with self.assertRaises(ZeroDivisionError):
             eval_metrics.ndcg(recommendations, ground_truth, k)

class TestCalculateAUC(unittest.TestCase):

    def test_perfect_ranking_is_one(self):
        y_true = [0, 0, 1, 1]
        y_prob = [0.1, 0.2, 0.8, 0.9]
        self.assertAlmostEqual(eval_metrics.auc(y_true, y_prob), 1.0)

    def test_worst_ranking_is_zero(self):
        y_true = [0, 0, 1, 1]
        y_prob = [0.8, 0.9, 0.1, 0.2]
        self.assertAlmostEqual(eval_metrics.auc(y_true, y_prob), 0.0)

    def test_random_ranking_is_half(self):
        y_true = [0, 1, 0, 1]
        y_prob = [0.2, 0.3, 0.6, 0.5]
        self.assertAlmostEqual(eval_metrics.auc(y_true, y_prob), 0.5)

    def test_nontrivial_ranking(self):
        y_true = [1, 0, 1, 0]
        y_prob = [0.9, 0.1, 0.3, 0.5]
        self.assertAlmostEqual(eval_metrics.auc(y_true, y_prob), 0.75)
        
    def test_ranking_with_tied_probabilities(self):
        y_true = [0, 1, 0, 1]
        y_prob = [0.6, 0.8, 0.6, 0.4]
        self.assertAlmostEqual(eval_metrics.auc(y_true, y_prob), 0.5)

    def test_tie_between_positive_and_negative(self):
        # This test breaks the original "fast_auc" implementation
        # as it does not handle probability ties correctly, it returns 0.75 here
        # if we were to switch the values at positions 0 and 1 in y_true "fast_auc" returns 1
        # In reality in cases like this this metric should return (1 + 0.75) / 2
        y_true = [1, 0, 1, 0]
        y_prob = [0.7, 0.7, 0.9, 0.2]
        self.assertAlmostEqual(eval_metrics.auc(y_true, y_prob), 0.875)

    def test_all_positives_raises_error(self):
        y_true = [1, 1, 1, 1]
        y_prob = [0.1, 0.2, 0.3, 0.4]
        with self.assertRaises(ValueError):
            eval_metrics.auc(y_true, y_prob)

    def test_all_negatives_raises_error(self):
        y_true = [0, 0, 0, 0]
        y_prob = [0.1, 0.2, 0.3, 0.4]
        with self.assertRaises(ValueError):
            eval_metrics.auc(y_true, y_prob)

class TestPropensityScoredPrecision(unittest.TestCase):

    def setUp(self):
        self.propensities = np.array([0.01, 0.1, 0.8, 0.9, 0.5])

    def _mpsp(self, ground_truth):
        return sum([1 / self.propensities[gt] for gt in ground_truth])

    def test_basic_case_with_mix_of_hits(self):
        recommendations = [1, 99, 2]
        ground_truth = {1, 2}
        k = 3
        
        upsp = ( 1 / 0.1 + 1 / 0.8 ) / 3
        mpsp = self._mpsp(ground_truth)
        psp = upsp / mpsp
        self.assertAlmostEqual(eval_metrics.psp(recommendations, ground_truth, self.propensities, k), psp)
    
    def test_uses_recommendation_len_when_lt_k(self):
        recommendations = [1, 99, 2]
        ground_truth = {1, 2}
        k = 5
        
        upsp = ( 1 / 0.1 + 1 / 0.8 ) / 3
        mpsp = self._mpsp(ground_truth)
        psp = upsp / mpsp
        self.assertAlmostEqual(eval_metrics.psp(recommendations, ground_truth, self.propensities, k), psp)

    def test_perfect_first_hit_on_niche_item(self):
        recommendations = [0, 99, 98]
        ground_truth = {0, 3}
        k = 1

        upsp = 1 / 0.01
        mpsp = self._mpsp(ground_truth)
        psp = upsp / mpsp
        self.assertAlmostEqual(eval_metrics.psp(recommendations, ground_truth, self.propensities, k), psp)

    def test_no_hits_in_top_k_is_zero(self):
        recommendations = [99, 98, 97]
        ground_truth = {0, 1, 2}
        k = 3

        self.assertAlmostEqual(eval_metrics.psp(recommendations, ground_truth, self.propensities, k), 0.0)

    def test_mpsp_denominator_is_independent_of_k(self):
        recommendations = [1, 99, 98]
        ground_truth = {1, 2}
        k = 1

        upsp = 1 / 0.1 
        mpsp = self._mpsp(ground_truth)
        self.assertAlmostEqual(eval_metrics.psp(recommendations, ground_truth, self.propensities, k), upsp / mpsp)

    def test_edge_case_empty_ground_truth(self):
        recommendations = [0, 1, 2]
        ground_truth = set()
        k = 3

        with self.assertRaises(ZeroDivisionError):
            eval_metrics.psp(recommendations, ground_truth, self.propensities, k)

    def test_edge_case_empty_recommendations(self):
        recommendations = []
        ground_truth = {0, 1}
        k = 3
        
        self.assertAlmostEqual(eval_metrics.psp(recommendations, ground_truth, self.propensities, k), 0.0)
    

class TestInterListJaccardDistance(unittest.TestCase):
    """Test cases for Inter-list Jaccard distance metric based on item tags."""
    
    def setUp(self):
        """Set up test data with various tag configurations."""
        # Sample item-to-tag mapping representing Steam-like game tags
        self.item_tag_mapping = {
            1: {"Action", "Shooter", "FPS"},
            2: {"Action", "Adventure", "RPG"},
            3: {"Strategy", "RTS", "Military"},
            4: {"Puzzle", "Casual", "Indie"},
            5: {"Action", "Shooter", "Multiplayer"},
            6: {"Racing", "Sports", "Simulation"},
            7: {"Horror", "Survival", "Action"},
            8: set(),  # Item with no tags
            9: {"Action"},  # Item with single tag
            10: {"Action", "Adventure", "RPG", "Strategy", "Shooter"}  # Item with many tags
        }
    
    def test_identical_items_zero_distance(self):
        """Items with identical tags should have zero Jaccard distance"""
        recommendations = [1, 1]  # Same item, identical tags
        k = 2
        result = eval_metrics.inter_list_jaccard_distance(recommendations, self.item_tag_mapping, k)
        self.assertAlmostEqual(result, 0.0)
    
    def test_completely_different_items_max_distance(self):
        """Items with no tag overlap should have distance 1.0"""
        recommendations = [3, 4]
        k = 2
        result = eval_metrics.inter_list_jaccard_distance(recommendations, self.item_tag_mapping, k)
        self.assertAlmostEqual(result, 1.0)
    
    def test_partial_overlap_items(self):
        """Items with partial tag overlap should have intermediate distance"""
        recommendations = [1, 2]
        k = 2
        # Intersection: {"Action"} = 1, Union: {"Action", "Shooter", "FPS", "Adventure", "RPG"} = 5
        # Jaccard similarity = 1/5 = 0.2, Jaccard distance = 1 - 0.2 = 0.8
        result = eval_metrics.inter_list_jaccard_distance(recommendations, self.item_tag_mapping, k)
        self.assertAlmostEqual(result, 0.8)

    def test_subset_relationship(self):
        """Items where one tag set is subset of another"""
        recommendations = [2, 10] 
        k = 2
        result = eval_metrics.inter_list_jaccard_distance(recommendations, self.item_tag_mapping, k)
        self.assertAlmostEqual(result, 2.0/5.0, places=5)

    def test_multiple_items_average_distance(self):
        """Test with more than 2 items - should return average of all pairwise distances"""
        recommendations = [1, 3, 4]  # Three completely different items
        k = 3
        # Pairwise distances: (1,3)=1.0, (1,4)=1.0, (3,4)=1.0
        # Average = 1.0
        result = eval_metrics.inter_list_jaccard_distance(recommendations, self.item_tag_mapping, k)
        self.assertAlmostEqual(result, 1.0)

    def test_items_with_no_tags(self):
        """Items with no tags should have distance 0.0"""
        recommendations = [8, 8]  # Both have empty tag sets
        k = 2
        result = eval_metrics.inter_list_jaccard_distance(recommendations, self.item_tag_mapping, k)
        self.assertAlmostEqual(result, 0.0)

    def test_one_item_no_tags_other_has_tags(self):
        """One item with no tags, other with tags"""
        recommendations = [1, 8]  # {"Action", "Shooter", "FPS"} vs {}
        k = 2
        # Intersection: {} = 0, Union: {"Action", "Shooter", "FPS"} = 3
        # Jaccard similarity = 0/3 = 0, Jaccard distance = 1.0
        result = eval_metrics.inter_list_jaccard_distance(recommendations, self.item_tag_mapping, k)
        self.assertAlmostEqual(result, 1.0)

    def test_single_item_returns_zero(self):
        """Single item cannot have inter-list distance"""
        recommendations = [1]
        k = 1
        result = eval_metrics.inter_list_jaccard_distance(recommendations, self.item_tag_mapping, k)
        self.assertAlmostEqual(result, 0.0)

    def test_empty_recommendations_returns_zero(self):
        """Empty recommendation list should return Warning and 0.0"""
        recommendations = []
        k = 5
        with self.assertWarns(UserWarning):
            result = eval_metrics.inter_list_jaccard_distance(recommendations, self.item_tag_mapping, k)
        self.assertAlmostEqual(result, 0.0)

    def test_k_larger_than_recommendations(self):
        """k larger than recommendation list should use all recommendations"""
        recommendations = [1, 5]
        k = 10
        result = eval_metrics.inter_list_jaccard_distance(recommendations, self.item_tag_mapping, k)
        self.assertAlmostEqual(result, 0.5)  # Complete different items

    def test_k_zero_returns_zero(self):
        """k=0 should return 0.0"""
        recommendations = [1, 2, 3]
        k = 0
        result = eval_metrics.inter_list_jaccard_distance(recommendations, self.item_tag_mapping, k)
        self.assertAlmostEqual(result, 0.0)

    def test_mixed_scenario_realistic(self):
        """More realistic scenario with mixed tag overlaps"""
        recommendations = [1, 2, 5, 9]  # Mix of overlapping and different items
        k = 4
        # Item tags:
        # 1: {"Action", "Shooter", "FPS"}
        # 2: {"Action", "Adventure", "RPG"}  
        # 5: {"Action", "Shooter", "Multiplayer"}
        # 9: {"Action"}
        # Pairwise distances:
        # (1,2): intersection={"Action"}=1, union=5, distance=1-1/5=0.8
        # (1,5): intersection={"Action","Shooter"}=2, union=4, distance=1-2/4=0.5
        # (1,9): intersection={"Action"}=1, union=3, distance=1-1/3=2/3
        # (2,5): intersection={"Action"}=1, union=5, distance=1-1/5=0.8
        # (2,9): intersection={"Action"}=1, union=3, distance=1-1/3=2/3
        # (5,9): intersection={"Action"}=1, union=3, distance=1-1/3=2/3
        # Average = (0.8 + 0.5 + 2/3 + 0.8 + 2/3 + 2/3) / 6
        expected = (0.8 + 0.5 + 2.0/3.0 + 0.8 + 2.0/3.0 + 2.0/3.0) / 6
        result = eval_metrics.inter_list_jaccard_distance(recommendations, self.item_tag_mapping, k)
        self.assertAlmostEqual(result, expected, places=5)

    def test_single_tag_items(self):
        """Items with single tags"""
        recommendations = [9, 9]  # Both have single tag {"Action"}
        k = 2
        result = eval_metrics.inter_list_jaccard_distance(recommendations, self.item_tag_mapping, k)
        self.assertAlmostEqual(result, 0.0)

class TestEntropy(unittest.TestCase):
    
    def test_entropy_perfect_uniform_distribution(self):
        """Test entropy with perfectly uniform distribution across categories."""
        # Equal distribution across 4 categories should give maximum normalized entropy
        category_counts = [25, 25, 25, 25]  # 100 recommendations evenly distributed
        expected_entropy = 1.0  # Normalized maximum entropy = log2(4)/log2(4) = 1.0
        result = eval_metrics.entropy(category_counts)
        self.assertAlmostEqual(result, expected_entropy, places=5)
    
    def test_entropy_single_category_dominance(self):
        """Test entropy when all recommendations are from one category."""
        # All recommendations from one category should give zero entropy
        category_counts = [100, 0, 0, 0]
        expected_entropy = 0.0
        result = eval_metrics.entropy(category_counts)
        self.assertEqual(result, expected_entropy)
    
    def test_entropy_two_categories_equal(self):
        """Test entropy with equal distribution across two categories."""
        category_counts = [50, 50, 0, 0]
        expected_entropy = 1.0  # Normalized: log2(2)/log2(2) = 1.0
        result = eval_metrics.entropy(category_counts)
        self.assertAlmostEqual(result, expected_entropy, places=5)
    
    def test_entropy_skewed_distribution(self):
        """Test entropy with skewed distribution favoring one category."""
        category_counts = [80, 10, 5, 5]  # Heavily skewed toward first category
        # Calculate expected normalized entropy manually
        total = sum(category_counts)
        probs = [count/total for count in category_counts if count > 0]
        raw_entropy = -sum(p * np.log2(p) for p in probs)
        expected_entropy = raw_entropy / np.log2(len(probs))  # Normalize by log2(4)
        
        result = eval_metrics.entropy(category_counts)
        self.assertAlmostEqual(result, expected_entropy, places=5)
    
    def test_entropy_with_zeros(self):
        """Test entropy calculation when some categories have zero recommendations."""
        category_counts = [60, 0, 40, 0, 0]
        # Only two categories have recommendations
        total = 100
        p1, p2 = 60/total, 40/total
        raw_entropy = -(p1 * np.log2(p1) + p2 * np.log2(p2))
        expected_entropy = raw_entropy / np.log2(2)  # Normalize by log2(2)
        
        result = eval_metrics.entropy(category_counts)
        self.assertAlmostEqual(result, expected_entropy, places=5)
    
    def test_entropy_single_recommendation(self):
        """Test entropy when there's only one recommendation."""
        category_counts = [1, 0, 0]
        expected_entropy = 0.0  # No diversity with single item
        result = eval_metrics.entropy(category_counts)
        self.assertEqual(result, expected_entropy)
    
    def test_entropy_empty_input(self):
        """Test entropy with empty category counts."""
        category_counts = []
        with self.assertRaises((ValueError, ZeroDivisionError)):
            eval_metrics.entropy(category_counts)
    
    def test_entropy_all_zeros(self):
        """Test entropy when all category counts are zero."""
        category_counts = [0, 0, 0, 0]
        with self.assertRaises((ValueError, ZeroDivisionError)):
            eval_metrics.entropy(category_counts)
    
    def test_entropy_negative_counts(self):
        """Test entropy with negative counts (should raise error)."""
        category_counts = [10, -5, 20]
        with self.assertRaises(ValueError):
            eval_metrics.entropy(category_counts)
    
    def test_entropy_float_counts(self):
        """Test entropy with float counts (should work)."""
        category_counts = [10.5, 20.3, 15.2]
        total = sum(category_counts)
        probs = [count/total for count in category_counts]
        raw_entropy = -sum(p * np.log2(p) for p in probs)
        expected_entropy = raw_entropy / np.log2(len(probs))  # Normalize by log2(3)
        
        result = eval_metrics.entropy(category_counts)
        self.assertAlmostEqual(result, expected_entropy, places=5)
    
    def test_entropy_large_number_of_categories(self):
        """Test entropy with many categories."""
        # 10 categories with equal distribution
        category_counts = [10] * 10
        expected_entropy = 1.0  # Normalized: log2(10)/log2(10) = 1.0
        result = eval_metrics.entropy(category_counts)
        self.assertAlmostEqual(result, expected_entropy, places=5)
    
    def test_entropy_monotonicity(self):
        """Test that entropy increases as distribution becomes more uniform."""
        # More skewed distribution
        skewed_counts = [90, 5, 3, 2]
        # Less skewed distribution  
        less_skewed_counts = [60, 20, 15, 5]
        # Even more uniform
        uniform_counts = [30, 25, 25, 20]
        
        skewed_entropy = eval_metrics.entropy(skewed_counts)
        less_skewed_entropy = eval_metrics.entropy(less_skewed_counts)
        uniform_entropy = eval_metrics.entropy(uniform_counts)
        
        # More uniform distributions should have higher entropy
        self.assertLess(skewed_entropy, less_skewed_entropy)
        self.assertLess(less_skewed_entropy, uniform_entropy)


if __name__ == '__main__':
    unittest.main()