from library.Neuron import *
from library.NeuralNetwork import *
import random


# Simulates a NeuralNetwork composed by only one Perceptron
class Perceptron(NeuralNetwork):

    # Initial Setup
    def __init__(self, sigmoid, n_weights, learn_rate, max_init_value):
        super().__init__(1)
        self.learn_rate = learn_rate

        # Set weights with random values based on the maximum specified value
        weights = []
        for i in range(0, n_weights):
            weights.append(random.uniform(0.0, max_init_value))

        # Defines the only layer with the only Neuron
        self.set_layer(0, [Neuron(weights, sigmoid)])

    def learn(self, error, inputs):
        # Updates values for each weight
        weights = self.layers[0][0].weights
        for i in range(0, len(weights)):
            # Applies delta rule for given weight
            weights[i] = weights[i] + self.learn_rate * error * inputs[i]

        # Saves new weights
        self.layers[0][0].weights = weights

        # Remake matrices for future calculations
        self.make_matrix()

    def train(self, train_data, labels):
        # Checks for parameters errors
        if len(train_data) != len(labels):
            raise ValueError("Train and Label datasets must have equal elements count!!")

        # Train for all samples in the training dataset
        for i in range(0, len(train_data)):
            # Gets result
            result = self.process(train_data[i]).tolist()[0][0]

            # Update weights if result is different from the expected
            if result != labels[i]:
                # Calculates errors
                error = labels[i] - result
                # Learn from error
                self.learn(error, train_data[i])
