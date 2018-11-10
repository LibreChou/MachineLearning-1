from library.NeuralNetwork import *
import random


class Perceptron(NeuralNetwork):

    def __init__(self, sigmoid, n_weights, learn_rate, max_init_value):
        super().__init__(1)

        self.learn_rate = learn_rate
        weights = []
        for i in range(0, n_weights):
            weights.append(random.uniform(0.0, max_init_value))

        self.set_layer(0, [Neuron(weights, sigmoid)])

    def get_error(self, data, label):
        result = self.process(data)[0]
        return label - result

    def learn(self, error, inputs):

        weights = self.layers[0][0].weights

        for i in range(0, len(weights)):
            weights[i] = weights[i] + self.learn_rate * error * inputs[i]

        self.layers[0][0].weights = weights
        self.make_matrix()

    def train(self, train_data, labels):

        if len(train_data) != len(labels):
            raise ValueError("Train label")

        for i in range(0, len(train_data)):
            result = self.process(train_data[i])[0]
            if result != labels[i]:
                error = labels[i] - result
                self.learn(error, train_data[i])


