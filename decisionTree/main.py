import unittest
import grimoire as g
import streamlit as st
"""
# Gini index

Gives an idea of how good the split is by how mixed the classes are in the two groups created by the split.

A perfect separation results in gini score of 0

"""


def gini_index(groups, classes):
    n_instances = sum([len(group) for group in groups])

    gini = 0

    for group in groups:
        size = len(group)

        if size == 0:
            continue

        score = 0
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p

        gini += (1.0 - score) * (size / n_instances)

    return gini


class GiniTest(unittest.TestCase):
    def test_worst_split_50_each_side(self):
        groups = [
            [[1, 1], [1, 0]],
            [[1, 1], [1, 0]]
        ]
        self.assertEqual(0.5, gini_index(groups, [0, 1]))

    def test_perfect_separation(self):
        groups = [
            [[1, 0], [1, 0]],
            [[1, 1], [1, 1]]
        ]
        self.assertEqual(0.0, gini_index(groups, [0, 1]))


if __name__ == '__main__':
    unittest.main()
