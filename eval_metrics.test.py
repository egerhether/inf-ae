import unittest
import numpy as np

import eval_metrics

class TestCalculateHitRate(unittest.TestCase):

    def test_no_hits(self):
        recommendations = [101, 102, 103]
        ground_truth = {1, 2, 3}
        k = 3
        self.assertAlmostEqual(eval_metrics.calculate_hit_rate(recommendations, ground_truth, k), 0.0)

    def test_2_hits_but_1_in_k(self):
        recommendations = [1, 2, 3, 4, 5]
        ground_truth = {3, 5, 10}
        k = 4
        self.assertAlmostEqual(eval_metrics.calculate_hit_rate(recommendations, ground_truth, k), 1/3)

    def test_all_gt_in_k(self):
        recommendations = [10, 20, 30, 40, 50]
        ground_truth = {20, 40}
        k = 5
        self.assertAlmostEqual(eval_metrics.calculate_hit_rate(recommendations, ground_truth, k), 1.0)

    def test_no_hits_in_k(self):
        recommendations = [1, 2, 3, 4, 5]
        ground_truth = {4, 5, 6}
        k = 2
        self.assertAlmostEqual(eval_metrics.calculate_hit_rate(recommendations, ground_truth, k), 0.0)

    def test_k_is_zero(self):
        recommendations = [1, 2, 3]
        ground_truth = {1, 2}
        k = 0
        self.assertAlmostEqual(eval_metrics.calculate_hit_rate(recommendations, ground_truth, k), 0.0)

    def test_k_is_larger_than_recommendation_list(self):
        recommendations = [1, 2]
        ground_truth = {1, 3}
        k = 10
        self.assertAlmostEqual(eval_metrics.calculate_hit_rate(recommendations, ground_truth, k), 0.5)

    def test_empty_recommendation_list(self):
        recommendations = []
        ground_truth = {1, 2, 3}
        k = 5
        self.assertAlmostEqual(eval_metrics.calculate_hit_rate(recommendations, ground_truth, k), 0.0)

    def test_empty_ground_truth_raises_error(self):
        recommendations = [1, 2, 3]
        ground_truth = set()
        k = 3
        with self.assertRaises(ZeroDivisionError):
             eval_metrics.calculate_hit_rate(recommendations, ground_truth, k)

class Testeval_CalculateNDCG(unittest.TestCase):

    def test_perfect_ranking_is_one(self):
        recommendations = [1, 2, 3, 4, 5]
        ground_truth = {1, 2, 3}
        k = 3
        self.assertAlmostEqual(eval_metrics.calculate_ndcg(recommendations, ground_truth, k), 1.0)

    def test_no_hits_is_zero(self):
        recommendations = [101, 102, 103]
        ground_truth = {1, 2, 3}
        k = 3
        self.assertAlmostEqual(eval_metrics.calculate_ndcg(recommendations, ground_truth, k), 0.0)

    def test_typical_good_ranking(self):
        recommendations = [1, 99, 2, 98, 3] # Hits at pos 1, 3, 5
        ground_truth = {1, 2, 3}
        k = 5
        expected_dcg = 1 / np.log2(1 + 1) + 1 / np.log2(3 + 1) + 1 / np.log2(5 + 1)
        expected_idcg = 1 / np.log2(1 + 1) + 1 / np.log2(2 + 1) + 1 / np.log2(3 + 1)
        expected_ndcg = expected_dcg / expected_idcg
        self.assertAlmostEqual(eval_metrics.calculate_ndcg(recommendations, ground_truth, k), expected_ndcg)

    def test_k_is_smaller_than_number_of_hits(self):
        recommendations = [1, 2, 3, 4, 5]
        ground_truth = {1, 2, 3, 4, 5}
        k = 3
        self.assertAlmostEqual(eval_metrics.calculate_ndcg(recommendations, ground_truth, k), 1.0)
        
    def test_empty_recommendations_is_zero(self):
        recommendations = []
        ground_truth = {1, 2, 3}
        k = 5
        self.assertAlmostEqual(eval_metrics.calculate_ndcg(recommendations, ground_truth, k), 0.0)

    def test_k_is_zero_raises_error(self):
        recommendations = [1, 2, 3]
        ground_truth = {1, 2}
        k = 0
        with self.assertRaises(ZeroDivisionError):
             eval_metrics.calculate_ndcg(recommendations, ground_truth, k)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)