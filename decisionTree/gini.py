import unittest
import grimoire as g
import grimoire.test as t
import streamlit as st

"""
# Gini index

The goal is to have perfect separatio between groups.
So one class is all in one group and the other is all in the other group.

This perfect scenario the return value is zero.
It keeps increasing until the worse case scenario 0.5

"""


def gini_index(groups, classes=[0,1]):

    total_population = g.reduce(lambda y, x: len(x) + y, groups)

    gini = 0

    for group in groups:
        group_population = len(group)

        if group_population == 0:
            continue

        group_score = 0
        for class_val in classes:
            percentage_in_group_with_class = group.count(class_val) / group_population
            group_score += percentage_in_group_with_class**2

        percentage_of_total_population = group_population / total_population

        gini += (1.0 - group_score) * percentage_of_total_population

    return gini


class GiniTest(unittest.TestCase):
    def test_perfect_separation(self):
        groups = [
            [0, 0],
            [1, 1]
        ]
        self.assertEqual(0.0, gini_index(groups))

    def test_unbalanced_groups(self):
        groups = [
            [0, 0],
            []
        ]
        self.assertEqual(0.0, gini_index(groups))

    def test_medium_case(self):
        groups = [
            [1, 1, 0],
            [1, 0, 0]
        ]
        self.assertEqual(0.44, round(gini_index(groups), 2))

        groups = [
            [1, 1, 1, 0],
            [1, 0, 0, 0]
        ]
        self.assertEqual(0.38, round(gini_index(groups), 2))


    def test_worst_split_50_each_side(self):
        groups = [
            [1, 0],
            [1, 0]
        ]
        self.assertEqual(0.5, gini_index(groups))

#t.run_test_case(GiniTest)

