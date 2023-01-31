import random
from math import exp

# Feedforward neural network


class Neuron:
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

    def transfer_derivative(self, output):
        return output * (1.0 - output)

    def forward_pass(self, input):
        self.output = self.transfer(self.activation(input))
        return self.output  # TODO: Something wrong?

    def backward_pass(self, expected_output):
        error = self.output - expected_output
        self.delta = error * self.transfer_derivative(self.output)
        print(self.delta)


class Layer:
    def __init__(self, size):
        self.neurons = []
        for _ in range(size):
            self.neurons.append(Neuron())

    def __eq__(self, other):
        return self.neurons == other.neurons

    def __str__(self):
        output = ""
        for neuron in self.neurons:
            output += f'{str(neuron)}\t'
        return output[:-1]

    def initialize_random(self, input_size):
        for neuron in self.neurons:
            neuron.initialize_random(input_size=input_size)

    def load_weights(self, neuron_weights):
        for neuron, weights in zip(self.neurons, neuron_weights):
            neuron.load_weights(weights)

    def forward_pass(self, input):
        output = []
        for neuron in self.neurons:
            output.append(neuron.forward_pass(input=input))
        return output

    def backward_pass(self, expected_outputs):
        for neuron, expected_output in zip(self.neurons, expected_outputs):
            neuron.backward_pass(expected_output=expected_output)


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

    def initialize_random(self, input_size, layer_sizes):
        self.layers = []
        for layer_size in layer_sizes:
            layer = Layer(size=layer_size)
            layer.initialize_random(input_size=input_size)
            self.layers.append(layer)
            input_size = layer_size

    def load_weights(self):
        self.layers = []
        with open(self.file_name, 'r') as f:
            for layer_weights in f.readlines():
                layer_weights = layer_weights[:-1].split('\t')
                layer = Layer(size=len(layer_weights))
                neuron_weights = [[float(x) for x in neuron.split(' ')]
                                  for neuron in layer_weights]
                layer.load_weights(neuron_weights=neuron_weights)
                self.layers.append(layer)

    def save_weights(self):
        with open(self.file_name, 'w') as f:
            f.writelines([f"{str(layer)}\n" for layer in self.layers])

    def forward_pass(self, input):
        for layer in self.layers:
            input = layer.forward_pass(input)
        return input

    def backward_pass(self, expected_outputs):
        for layer in reversed(self.layers):
            expected_outputs = layer.backward_pass(expected_outputs)


# nn = NeuralNetwork(file_name="test_weights/test2")
# nn.load_weights()
# nn.forward_pass([1, 0])
# nn.backward_pass(expected_outputs=[0, 1])
