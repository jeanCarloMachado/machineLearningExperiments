from random import random, seed, randrange
from math import exp
from csv import reader


def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()

    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]

    network.append(hidden_layer)

    output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)

    return network


def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]

    return activation


def transfer(activation):
    # transform the actication into a sigmoid
    return 1.0 / (1.0 + exp(-activation))


def forward_propagate(network, row):
    # from input to a network output

    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs

    return inputs


def transfer_derivative(output):
    # calculate the derivative of an neuron output
    return output * (1.0 - output)


def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])

        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]

        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]

        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']


def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0

        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i] - outputs[i])**2 for i in range(len(expected))])

            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)

        print(f"> epoch={epoch}, lrate={l_rate}, error={sum_error}")


def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))



def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader  = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i

    for row in dataset:
        row[column] = lookup[row[column]]

    return lookup


def dataset_minmax(dataset):
    stats = [[min(column), max(column)] for column in zip(*dataset)]

    return stats


def noramlize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row) -1):
            row[i] = (row[i] - minmax[i][0] / (minmax[i][1] - minmax[i][0]))


def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)

    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))

        dataset_split.append(fold)
        return dataset_split


def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1

    return correct / float(len(actual)) * 100.0


def evaluate_algorithm(dataset, algorithm, n_folds, *arags):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()

    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()

        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None

        predicted = algorithm(train_set< test_set, *args)
        actual=  [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)

    return scores


"""
test below
"""


def test_predict():
    dataset = [[2.7810836, 2.550537003, 0], [1.465489372, 2.362125076, 0],
               [3.396561688, 4.400293529, 0], [1.38807019, 1.850220317, 0],
               [3.06407232, 3.005305973, 0], [7.627531214, 2.759262235, 1],
               [5.332441248, 2.088626775, 1], [6.922596716, 1.77106367, 1],
               [8.675418651, -0.242068655, 1], [7.673756466, 3.508563011, 1]]
    network = [[{
        'weights': [-1.482313569067226, 1.8308790073202204, 1.078381922048799]
    }, {
        'weights': [0.23244990332399884, 0.3621998343835864, 0.40289821191094327]
    }],
               [{
                   'weights': [2.5001872433501404, 0.7887233511355132, -1.1026649757805829]
               }, {
                   'weights': [-2.429350576245497, 0.8357651039198697, 1.0699217181280656]
               }]]

    for row in dataset:
        prediction = predict(network, row)
        print(f"Expected={row[-1]},  Got={prediction}")





def test_train_network():
    seed(1)
    dataset = [[2.7810836, 2.550537003, 0], [1.465489372, 2.362125076, 0],
               [3.396561688, 4.400293529, 0], [1.38807019, 1.850220317, 0],
               [3.06407232, 3.005305973, 0], [7.627531214, 2.759262235, 1],
               [5.332441248, 2.088626775, 1], [6.922596716, 1.77106367, 1],
               [8.675418651, -0.242068655, 1], [7.673756466, 3.508563011, 1]]
    n_inputs = len(dataset[0]) - 1
    n_outputs = len(set(row[-1] for row in dataset))
    network = initialize_network(n_inputs, 2, n_outputs)
    train_network(network, dataset, 0.5, 20, n_outputs)
    for layer in network:
        print(layer)


def test_back_propagation():
    network = [[{
        'output': 0.7105668883115941,
        'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]
    }],
               [{
                   'output': 0.6213859615555266,
                   'weights': [0.2550690257394217, 0.49543508709194095]
               }, {
                   'output': 0.6573693455986976,
                   'weights': [0.4494910647887381, 0.651592972722763]
               }]]
    expected = [0, 1]
    backward_propagate_error(network, expected)
    for layer in network:
        print(layer)


def test_forward_propagation():
    network = [[{
        'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]
    }],
               [{
                   'weights': [0.2550690257394217, 0.49543508709194095]
               }, {
                   'weights': [0.4494910647887381, 0.651592972722763]
               }]]

    row = [1, 0, None]
    output = forward_propagate(network, row)
    print(output)


def test_initialize():
    seed(1)
    net = initialize_network(2, 1, 2)

    for layer in net:
        print(layer)
