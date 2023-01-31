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

    def size(self):
        return len(self.weights)

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
        try:
            return 1.0 / (1.0 + exp(-activation))
        except OverflowError:
            return float('inf')

    def transfer_derivative(self, output):
        return output * (1.0 - output)

    def forward_pass(self, input):
        self.output = self.transfer(self.activation(input))
        return self.output

    def delta_error(self, expected_output):
        error = self.output - expected_output
        self.delta = error * self.transfer_derivative(self.output)

    def backward_pass(self, index, next_layer):
        self.delta = 0
        for neuron in next_layer.neurons:
            self.delta += neuron.weights[index] * neuron.delta

    def update_weights(self, prev_layer_outputs, learning_rate):
        for i, prev_layer_output in enumerate(prev_layer_outputs):
            self.weights[i] -= learning_rate * \
                self.delta * prev_layer_output
        self.weights[-1] -= learning_rate * self.delta  # Bias


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

    def size(self):
        return [neuron.size() for neuron in self.neurons]

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

    def last_layer_error(self, expected_outputs):
        for neuron, expected_output in zip(self.neurons, expected_outputs):
            neuron.delta_error(expected_output=expected_output)

    def backward_pass(self, next_layer):
        for i, neuron in enumerate(self.neurons):
            neuron.backward_pass(index=i, next_layer=next_layer)

    def update_weights(self, prev_layer_outputs, learning_rate):
        for neuron in self.neurons:
            neuron.update_weights(prev_layer_outputs=prev_layer_outputs,
                                  learning_rate=learning_rate)


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

    def size(self):
        return [layer.size() for layer in self.layers]

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

    def forward_pass(self, inputs):
        for layer in self.layers:
            inputs = layer.forward_pass(inputs)
        return inputs

    def backward_pass(self, expected_outputs):
        last_layer = self.layers[-1]
        last_layer.last_layer_error(expected_outputs=expected_outputs)
        for layer in reversed(self.layers[:-1]):
            layer.backward_pass(next_layer=last_layer)
            last_layer = layer

    def update_weights(self, inputs, learning_rate):
        prev_layer_outputs = inputs
        for layer in self.layers:
            layer.update_weights(prev_layer_outputs=prev_layer_outputs,
                                 learning_rate=learning_rate)
            prev_layer_outputs = [neuron.output for neuron in layer.neurons]

    def train(self, training_data, training_data_outputs, epochs, learning_rate):
        for epoch in range(epochs):
            sum_error = 0
            for inputs, expected_outputs in zip(training_data, training_data_outputs):
                outputs = self.forward_pass(inputs=inputs)
                sum_error += sum([(expected_outputs[i] - outputs[i])
                                 ** 2 for i in range(len(expected_outputs))])
                self.backward_pass(expected_outputs=expected_outputs)
                self.update_weights(inputs=inputs, learning_rate=learning_rate)
            print(
                f'Epoch: {epoch}, l_rate: {learning_rate}, error: {sum_error}')

    def predict(self, inputs):
        return self.forward_pass(inputs=inputs)
