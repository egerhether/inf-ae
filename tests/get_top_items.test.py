import sys
import os
import unittest
import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cold_start import get_popular_items, get_popular_diverse_items

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

class TestSimpleDiversePopularity(unittest.TestCase):
    
    def test_round_robin_selection(self):
        # We have 5 items in two categories: 'Action' and 'Comedy'.
        # 'Action' is the more popular category overall.
        dense_matrix = np.array([
            # Item: 0  1  2 | 3  4
            #      (Action) | (Comedy)
            [1, 1, 0, 1, 0],  # User 0
            [1, 0, 0, 1, 0],  # User 1
            [1, 1, 0, 0, 0],  # User 2
            [1, 0, 1, 0, 0],  # User 3
        ])
        
        # Resulting popularities:
        # Item 0 (Action): 4
        # Item 1 (Action): 2
        # Item 2 (Action): 1
        # Item 3 (Comedy): 2
        # Item 4 (Comedy): 0
        
        train_matrix = csr_matrix(dense_matrix)
        
        item_categories = {
            0: 'Action', 1: 'Action', 2: 'Action',
            3: 'Comedy', 4: 'Comedy'
        }
        
        k = 3
        
        actual_items = get_popular_diverse_items(
            train_matrix=train_matrix,
            item_tag_mapping=item_categories,
            k=k
        )
        
        # Logic Trace:
        # - Category 'Action' popularity = 4+2+1 = 7.
        # - Category 'Comedy' popularity = 2+0 = 2.
        # - Sorted categories: ['Action', 'Comedy']
        # - Round 1, pick from 'Action': Item 0 (most popular action). Items: [0]
        # - Round 1, pick from 'Comedy': Item 3 (most popular comedy). Items: [0, 3]
        # - Round 2, pick from 'Action': Item 1 (second most popular action). Items: [0, 3, 1]
        
        expected_items = [0, 3, 1]
        
        self.assertEqual(actual_items, expected_items,)

if __name__ == '__main__':
    unittest.main()