from random import random, seed
from math import exp

def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()

    hidden_layer= [
        {'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)
    ]

    network.append(hidden_layer)

    output_layer = [
        {'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)
    ]
    network.append(output_layer)


    return network


def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]

    return activation


def transfer(activation):
    #transform the actication into a sigmoid 
    return 1.0 / (1.0 + exp(-activation))


def forward_propagate(network, row):
    #from input to a network output

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
    #calculate the derivative of an neuron output
    return output * (1.0 - output)


def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i+1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron= layer[j]
                errors.append(expected[j] - neuron['output'])

        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


"""
test below
"""


def test_back_propagation():
    network = [[{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}], [{'output': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095]}, {'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763]}]] 
    expected = [0, 1]
    backward_propagate_error(network, expected)
    for layer in network:
        print(layer)



def test_forward_propagation():
    network = [[{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}], [{'weights': [0.2550690257394217, 0.49543508709194095]}, {'weights': [0.4494910647887381, 0.651592972722763]}]] 

    row = [1, 0, None]
    output = forward_propagate(network, row)
    print(output)


def test_initialize():
    seed(1)
    net = initialize_network(2, 1, 2)

    for layer in net:
        print(layer)
