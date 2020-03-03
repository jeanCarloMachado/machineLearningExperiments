
import logging
import unittest
"""
activation = sum(weight_i * x_i ) + bias

prediction = 1.0 if activation >= 0.0 else 0.0

Each iteration of the weights are updated using the equation below:

w = w + learning_rate * (expected - predicted) * x

"""

class Perceptron:
    @staticmethod
    def predict(row, weights):
        activation = weights[0]

        for i in range(len(row)-1):
            activation += weights[i + 1] * row[i]

        logging.debug(f'Activation = {activation}')

        return 1.0 if activation >= 0.0 else 0.0

    @staticmethod
    def train_weights(train, l_rate, n_epoch):
        weights = [0.0 for i in range(len(train[0]))]
        for epoch in range(n_epoch):
            sum_error = 0.0
            for row in train:
                prediction = predict(row, weights)
                error = row[-1] - prediction
                sum_error += error**2
                weights[0] = weights[0] + l_rate * error

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
        dataset = [[2.7810836, 2.550537003, 0],
                   [1.465489372, 2.362125076, 0],
                   [3.396561688, 4.400293529, 0],
                   [1.38807019, 1.850220317, 0],
                   [3.06407232, 3.005305973, 0],
                   [7.627531214, 2.759262235, 1],
                   [5.332441248, 2.088626775, 1],
                   [6.922596716, 1.77106367, 1],
                   [8.675418651, -0.242068655, 1],
                   [7.673756466, 3.508563011, 1]]
        weights = [-0.1, 0.20653640140000007, -0.23418117710000003]

        for row in dataset:
            prediction = Perceptron.predict(row, weights)
            print(f"Expected={row[-1]}, Predicted={prediction}")



def tests():
    suite = unittest.TestSuite()
    suite.addTest(Validate("test_predict"))
    runner = unittest.TextTestRunner()
    runner.run(suite)
