import unittest
import numpy as np
import jax.numpy as jnp
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cold_start import get_recommendations

class TestGetRecommendations(unittest.TestCase):

    def test_very_cold_recommendation(self):
        logits = jnp.array([
            [0.1, 0.2, 0.8, 0.4, 0.5],  # User 0
            [0.9, 0.7, 0.6, 0.5, 0.4],  # User 1
        ])
        input_items = [[], []]
        k = 3
        
        expected = [
            [2, 4, 3],  # User 0: scores 0.8, 0.5, 0.4
            [0, 1, 2],  # User 1: scores 0.9, 0.7, 0.6
        ]
        
        result = get_recommendations(logits, input_items, k)
        self.assertEqual(result, expected)

    def test_exclude_seen_items(self):
        logits = jnp.array([
            [0.1, 0.2, 0.8, 0.4, 0.9],  # User 0
            [0.9, 0.7, 0.6, 0.5, 0.4],  # User 1
        ])
        # User 0 has seen item 4 (highest score)
        # User 1 has seen item 0 (highest score)
        input_items = [[4], [0]] 
        k = 2
        
        expected = [
            [2, 3],  # User 0: next best are 2 (0.8) and 3 (0.4)
            [1, 2],  # User 1: next best are 1 (0.7) and 2 (0.6)
        ]
        
        result = get_recommendations(logits, input_items, k)
        self.assertEqual(result, expected)

    def test_fewer_than_k_available(self):
        logits = jnp.array([[0.1, 0.2, 0.8, 0.4, 0.5]])
        input_items = [[2, 4, 3]]
        k = 3
        expected = [
            [1, 0] # Recommend item 1 (0.2) and item 0 (0.1)
        ]

        result = get_recommendations(logits, input_items, k)
        self.assertEqual(result, expected)

    def test_all_items_seen(self):
        logits = jnp.array([
            [0.1, 0.2, 0.8, 0.4, 0.5]
        ])
        input_items = [[0, 1, 2, 3, 4]] # All items seen
        k = 3
        
        expected = [[]]
        
        result = get_recommendations(logits, input_items, k)
        self.assertEqual(result, expected)

    def test_empty_input(self):
        logits = jnp.empty((0, 5)) # No users
        input_items = []
        k = 3
        
        expected = []
        
        result = get_recommendations(logits, input_items, k)
        self.assertEqual(result, expected)

    def test_k_equals_zero(self):
        logits = jnp.array([
            [0.1, 0.2, 0.8, 0.4, 0.5],
            [0.9, 0.7, 0.6, 0.5, 0.4],
        ])
        input_items = [[], []]
        k = 0

        with self.assertRaises((ValueError)): 
            get_recommendations(logits, input_items, k)

if __name__ == '__main__':
    unittest.main()