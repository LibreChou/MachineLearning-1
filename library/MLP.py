import numpy as np
from library.NeuralNetwork import *
from library.Neuron import *
import random


class MLP(NeuralNetwork):

    # Initial setup
    def __init__(self, learn_rate, max_init_value, layers_number):
        super().__init__(layers_number)
        self.learn_rate = learn_rate
        self.max_init_value = max_init_value

    def learn_for_layer(self, error, inputs, layer):
        print("hi")

    def learn(self, error, inputs):
        error_mx = [None] * self.layers_number

        # Error for exposed layer
        error_mx[self.layers_number - 1] = []

        for i in range(0, len(error)):
            error_mx[self.layers_number - 1].append(error[i] * self.layers_output[self.layers_number - 1][i][0] *
                                                    (1 - self.layers_output[self.layers_number - 1][i][0]))

        # Calculates error
        for i in range(self.layers_number - 2, -1, -1):
            error_mx[i] = []
            for j in range(0, len(self.layers[i])):
                sum = 0.0
                for k in range(0, len(self.layers[i+1])):
                    sum += error_mx[i + 1][k] * self.layers[i + 1][k].weights[j]
                sum = sum * self.layers_output[i][j][0] * (1 - self.layers_output[i][j][0])
                error_mx[i].append(sum)

        # Calculates new weights for all except first layer
        for i in range(self.layers_number - 1, -1, -1):
            for j in range(0, len(self.layers[i])):
                for k in range(0, len(self.layers[i][j].weights)):
                    if i != 0:
                        delta = self.learn_rate * error_mx[i][j] * self.layers_output[i - 1][k][0]
                    else:
                        delta = self.learn_rate * error_mx[i][j] * inputs[k]

        print ("hi")

        return
        # Calculates error for hidden layers
        for i in range(self.layers_number - 2, -1, -1):
            new_error = []
            for j in range(0, self.layers_mx[i].shape[0]):
                sum = 0.0
                for k in range(0, self.layers_mx[i + 1].shape[1]):
                    sum += self.layers_mx[i+1].item(j, k) * error_mx[i + 1][j]
                new_error.append(sum * self.layers_output[i][j][0] * (1 - self.layers_output[i][j][0]))
            error_mx[i] = new_error

        # Calculates new Weights for all layers
        for i in range(self.layers_number - 1, -1, -1):
            for j in range(0, self.layers_mx[i].shape[0]):
                for k in range(0, len(self.layers[i][j].weights)):
                    if i == 0:
                        delta = self.learn_rate * error_mx[i][j] * inputs[j]
                    else:
                        delta = self.learn_rate * error_mx[i][j] * self.layers_output[i - 1][j][0] * self.layers[i][j].weights[k]
                    new_theta = self.layers[i][j].weights[k] + delta
                    self.layers[i][j].weights[k] = new_theta

        print("hi")

    def init_random_weights(self):
        for i in range(0, self.layers_number):
            for neuron in self.layers[i]:
                weights = []
                for j in range(0, len(neuron.weights)):
                    weights.append(random.uniform(0.0, self.max_init_value))
                neuron.weights = weights

    @staticmethod
    def calc_error(result, expected):
        if len(result) != len(expected):
            raise ValueError("Result and labels cannot have different length!!")

        error = []
        for i in range(0, len(result)):
            error.append(expected[i] - result[i])

        return error

    def train(self, train_data, labels, init_weights=False):
        # Checks for parameters errors
        if len(train_data) != len(labels):
            raise ValueError("Train and Label datasets must have equal elements count!!")

        # Set weights with random values based on the maximum specified value
        if init_weights:
            self.init_random_weights()

        self.make_matrix()

        # Train for all samples in the training dataset
        for i in range(0, len(train_data)):
            # Gets result
            result = self.process(train_data[i])

            # Update weights if result is different from the expected
            if result != labels[i]:
                # Calculates errors
                error = self.calc_error(result, labels[i])
                # Learn from error
                self.learn(error, train_data[i])

    def process(self, inputs):
        output = super().process(inputs).tolist()
        return [o[0] for o in output]

