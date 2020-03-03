import logging
import unittest
from typing import *
"""
activation = sum(weight_i * x_i ) + bias

prediction = 1.0 if activation >= 0.0 else 0.0

Each iteration of the weights are updated using the equation below:

w = w + learning_rate * (expected - predicted) * x


Tue 03 Mar 2020 08:55:50 PM CET
Continue implementation from cross_validation_split
"""


class Perceptron:
    @staticmethod
    def predict(row: List[float], weights: List[float]) -> float:
        weights_copy = weights.copy()
        activation = weights_copy.pop(0)

        columns_of_data = row[:-1]
        for index, column in enumerate(columns_of_data):
            activation += weights_copy[index] * column

        logging.debug(f'Activation = {activation}')

        return 1.0 if activation >= 0.0 else 0.0

    @staticmethod
    def train_weights(train: List[Any], l_rate: float, n_epoch: int) -> List[float]:
        weights = [0.0 for i in range(len(train[0]))]
        for epoch in range(n_epoch):
            sum_error = 0.0
            for row in train:
                prediction = Perceptron.predict(row, weights)
                error = row[-1] - prediction
                sum_error += error**2
                weights[0] = weights[0] + l_rate * error

                for i in range(len(row) - 1):
                    weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
            print(f'>epoch={epoch}, lrate={l_rate}, error={sum_error}, weights={weights}')

        return weights


class Validate(unittest.TestCase):
    def test_predict(self):
        row = [1, 2, 3, 4]
        weights = [0.5, 0.3, 0.2, 0.1]
        result = Perceptron.predict(row, weights)
        self.assertEqual(1.0, result)

        row = [-1, -1, -1, -1]
        weights = [1, 1, 1, 1]
        result2 = Perceptron.predict(row, weights)
        self.assertEqual(0.0, result2)

    def test_dataset(self):
        dataset = [[2.7810836, 2.550537003, 0], [1.465489372, 2.362125076, 0],
                   [3.396561688, 4.400293529, 0], [1.38807019, 1.850220317, 0],
                   [3.06407232, 3.005305973, 0], [7.627531214, 2.759262235, 1],
                   [5.332441248, 2.088626775, 1], [6.922596716, 1.77106367, 1],
                   [8.675418651, -0.242068655, 1], [7.673756466, 3.508563011, 1]]

        weights = [-0.1, 0.20653640140000007, -0.23418117710000003]

        for row in dataset:
            prediction = Perceptron.predict(row, weights)
            print(f"Expected={row[-1]}, Predicted={prediction}")
            self.assertEqual(row[-1], prediction)

    def test_train_weights(self):
        train = [
            [0.5, 1.2, 0],
            [0.1, 2.2, 1],
        ]

        result = Perceptron.train_weights(train, 0.01, 5)
        self.assertEqual(result, [-0.01, -0.013000000000000001, 0.008000000000000004])

    def test_calculate_weights(self):
        dataset = [[2.7810836, 2.550537003, 0], [1.465489372, 2.362125076, 0],
                   [3.396561688, 4.400293529, 0], [1.38807019, 1.850220317, 0],
                   [3.06407232, 3.005305973, 0], [7.627531214, 2.759262235, 1],
                   [5.332441248, 2.088626775, 1], [6.922596716, 1.77106367, 1],
                   [8.675418651, -0.242068655, 1], [7.673756466, 3.508563011, 1]]
        l_rate = 0.1
        n_epoch = 5
        weights = Perceptron.train_weights(dataset, l_rate, n_epoch)
        print(weights)


def tests():
    suite = unittest.TestSuite()
    suite.addTest(Validate("test_predict"))
    suite.addTest(Validate("test_train_weights"))
    suite.addTest(Validate("test_calculate_weights"))
    suite.addTest(Validate("test_dataset"))
    runner = unittest.TextTestRunner()
    runner.run(suite)
