import random
from math import exp

# Feedforward neural network


class Node:
    def __init__(self):
        self.weights = []

    def __eq__(self, other):
        return self.weights == other.weights

    def __str__(self):
        output = ""
        for weight in self.weights:
            output += f'{str(weight)} '
        return output[:-1]

    def initialize_random(self, input_size):
        self.weights = []
        for _ in range(input_size):
            self.weights.append(random.random())
        self.weights.append(random.random())  # Bias

    def load_weights(self, weights):
        self.weights = weights

    def activation(self, input):
        assert len(input) == len(self.weights) - 1
        return sum([input[i] * self.weights[i] for i in range(len(input))]) + self.weights[-1]

    def transfer(self, activation):
        return 1.0 / (1.0 + exp(-activation))

    def forward_pass(self, input):
        return self.transfer(self.activation(input))


class Layer:
    def __init__(self, size):
        self.nodes = []
        for _ in range(size):
            self.nodes.append(Node())

    def __eq__(self, other):
        return self.nodes == other.nodes

    def __str__(self):
        output = ""
        for node in self.nodes:
            output += f'{str(node)}\t'
        return output[:-1]

    def initialize_random(self, input_size):
        for node in self.nodes:
            node.initialize_random(input_size=input_size)

    def load_weights(self, node_weights):
        for node, weights in zip(self.nodes, node_weights):
            node.load_weights(weights)

    def forward_pass(self, input):
        output = []
        for node in self.nodes:
            output.append(node.forward_pass(input=input))
        return output


class NeuralNetwork:
    def __init__(self, file_name="weights"):
        self.layers = []
        self.file_name = f'{file_name}.txt'

    def __eq__(self, other):
        return self.layers == other.layers

    def __str__(self):
        output = ""
        for i, layer in enumerate(self.layers):
            output += f'Layer {i}:\n{str(layer)}\n'
        return output[:-1]

    def initialize_random(self, sizes):
        self.layers = []
        input_size = sizes[0]
        for size in sizes[1:]:
            layer = Layer(size=size)
            layer.initialize_random(input_size=input_size)
            self.layers.append(layer)
            input_size = size

    def load_weights(self):
        self.layers = []
        with open(self.file_name, 'r') as f:
            for layer_weights in f.readlines():
                layer_weights = layer_weights[:-1].split('\t')
                layer = Layer(size=len(layer_weights))
                node_weights = [[float(x) for x in node.split(' ')]
                                for node in layer_weights]
                layer.load_weights(node_weights=node_weights)
                self.layers.append(layer)

    def save_weights(self):
        with open(self.file_name, 'w') as f:
            f.writelines([f"{str(layer)}\n" for layer in self.layers])

    def forward_pass(self, input):
        for layer in self.layers:
            input = layer.forward_pass(input)
        return input
