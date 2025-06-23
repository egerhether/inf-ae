import sys
import os
import unittest
import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cold_start import get_popular_items

class TestGetPopularItems(unittest.TestCase):
    
        
    def setUp(self):
        """
        Set up a sample sparse matrix.
        
        Popularity counts (sum of each column):
        Item 0: 1
        Item 1: 2
        Item 2: 3
        Item 3: 4
        Item 4: 2
        """
        # Verify the popularity counts by summing the columns of the dense matrix
        dense_matrix = np.array([
            # Item: 0  1  2  3  4
            [1, 0, 1, 1, 0],  # User 0
            [0, 1, 1, 1, 0],  # User 1
            [0, 1, 0, 1, 1],  # User 2
            [0, 0, 1, 1, 1],  # User 3
        ])
        item_counts = dense_matrix.sum(axis=0)
        assert list(item_counts) == [1, 2, 3, 4, 2]

        self.matrix = csr_matrix(dense_matrix)

    def test_basic_functionality(self):
        k = 2
        expected_ids = [3, 2]
        top_items = get_popular_items(self.matrix, k)
        self.assertEqual(expected_ids, top_items)

    def test_with_ties(self):
        k = 4
        top_items = get_popular_items(self.matrix, k)
        
        self.assertEqual(top_items[0], 3)
        self.assertEqual(top_items[1], 2)
        self.assertEqual(set(top_items[2:4]), {1, 4})

    def test_k_larger_than_num_items(self):
        num_items = self.matrix.shape[1]
        k = num_items + 5
        top_items = get_popular_items(self.matrix, k)
        
        expected_full_order = [3, 2, 4, 1, 0]
        self.assertEqual(top_items[0], expected_full_order[0])
        self.assertEqual(set(top_items[1:3]), {expected_full_order[1], expected_full_order[2]})
        self.assertEqual(top_items[3], expected_full_order[3])
        self.assertEqual(top_items[4], expected_full_order[4])

if __name__ == '__main__':
    unittest.main()