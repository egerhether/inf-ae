import unittest
import numpy as np

import eval_metrics

class TestCalculateHitRate(unittest.TestCase):

    def test_no_hits(self):
        recommendations = [101, 102, 103]
        ground_truth = {1, 2, 3}
        k = 3
        self.assertAlmostEqual(eval_metrics.hr(recommendations, ground_truth, k), 0.0)

    def test_2_hits_but_1_in_k(self):
        recommendations = [1, 2, 3, 4, 5]
        ground_truth = {3, 5, 10}
        k = 4
        self.assertAlmostEqual(eval_metrics.hr(recommendations, ground_truth, k), 1/3)

    def test_all_gt_in_k(self):
        recommendations = [10, 20, 30, 40, 50]
        ground_truth = {20, 40}
        k = 5
        self.assertAlmostEqual(eval_metrics.hr(recommendations, ground_truth, k), 1.0)

    def test_no_hits_in_k(self):
        recommendations = [1, 2, 3, 4, 5]
        ground_truth = {4, 5, 6}
        k = 2
        self.assertAlmostEqual(eval_metrics.hr(recommendations, ground_truth, k), 0.0)

    def test_k_is_zero(self):
        recommendations = [1, 2, 3]
        ground_truth = {1, 2}
        k = 0
        self.assertAlmostEqual(eval_metrics.hr(recommendations, ground_truth, k), 0.0)

    def test_k_is_larger_than_recommendation_list(self):
        recommendations = [1, 2]
        ground_truth = {1, 3}
        k = 10
        self.assertAlmostEqual(eval_metrics.hr(recommendations, ground_truth, k), 0.5)

    def test_empty_recommendation_list(self):
        recommendations = []
        ground_truth = {1, 2, 3}
        k = 5
        self.assertAlmostEqual(eval_metrics.hr(recommendations, ground_truth, k), 0.0)

    def test_empty_ground_truth_raises_error(self):
        recommendations = [1, 2, 3]
        ground_truth = set()
        k = 3
        with self.assertRaises(ZeroDivisionError):
             eval_metrics.hr(recommendations, ground_truth, k)

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
    

if __name__ == '__main__':
    unittest.main()