import unittest

from metrics import calculate_hit_rate


class TestCalculateHitRate(unittest.TestCase):


    def test_no_hits(self):
        recommendations = [101, 102, 103]
        ground_truth = {1, 2, 3}
        k = 3
        self.assertAlmostEqual(calculate_hit_rate(recommendations, ground_truth, k), 0.0)

    def test_2_hits_but_1_in_k(self):
        recommendations = [1, 2, 3, 4, 5]
        ground_truth = {3, 5, 10}
        k = 4
        self.assertAlmostEqual(calculate_hit_rate(recommendations, ground_truth, k), 1/3)

    def test_all_gt_in_k(self):
        recommendations = [10, 20, 30, 40, 50]
        ground_truth = {20, 40}
        k = 5
        self.assertAlmostEqual(calculate_hit_rate(recommendations, ground_truth, k), 1.0)

    def test_no_hits_in_k(self):
        recommendations = [1, 2, 3, 4, 5]
        ground_truth = {4, 5, 6}
        k = 2
        self.assertAlmostEqual(calculate_hit_rate(recommendations, ground_truth, k), 0.0)

    def test_k_is_zero(self):
        recommendations = [1, 2, 3]
        ground_truth = {1, 2}
        k = 0
        self.assertAlmostEqual(calculate_hit_rate(recommendations, ground_truth, k), 0.0)

    def test_k_is_larger_than_recommendation_list(self):
        recommendations = [1, 2]
        ground_truth = {1, 3}
        k = 10
        self.assertAlmostEqual(calculate_hit_rate(recommendations, ground_truth, k), 0.5)

    def test_empty_recommendation_list(self):
        recommendations = []
        ground_truth = {1, 2, 3}
        k = 5
        self.assertAlmostEqual(calculate_hit_rate(recommendations, ground_truth, k), 0.0)

    def test_empty_ground_truth_raises_error(self):
        recommendations = [1, 2, 3]
        ground_truth = set()
        k = 3
        with self.assertRaises(ZeroDivisionError):
             calculate_hit_rate(recommendations, ground_truth, k)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)