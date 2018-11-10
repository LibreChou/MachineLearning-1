
import numpy as np


# This class represents the structure of a generic multilayer neural network
class NeuralNetwork(object):

    # Initialize properties
    def __init__(self, layers_number):
        self.layers_number = layers_number
        self.layers = [None] * layers_number
        self.layers_mx = []

    # Sets a layer according to the defined neurons
    def set_layer(self, layer, neurons):

        # Creates new Layer
        new_layer = []
        for neuron in neurons:
            new_layer.append(neuron)

        # Creates layer
        self.layers[layer] = new_layer

    # Transforms Neuron's objects weights into a matrix
    def make_matrix(self):
        # Resets layers matrices
        self.layers_mx = [None] * self.layers_number

        # For all layers
        for i in range(0, self.layers_number):

            # Set each Neuron's weights for each line
            layer = []
            for neuron in self.layers[i]:
                layer.append(neuron.weights)

            # Stores matrix
            self.layers_mx[i] = np.matrix(layer)

    # Calculates the output for a given layer based on given input
    def calc_layer_output(self, layer, inputs):
        # Sets for mathmetical operation
        inputs = np.matrix(inputs).transpose()
        layer_mx = self.layers_mx[layer]

        # Calculates the output
        output = np.matmul(layer_mx, inputs)

        # Calculates the Fnet output for each neuron's output
        for i in range(0, output.shape[0]):
            value = self.layers[layer][i].fnet(output.item((i, 0)))
            output.itemset((i, 0), value)

        return output

    # Process the inputs to get the Neural Network's output
    def process(self, inputs):
        # Calculates matrices if needed
        if not self.layers_mx:
            self.make_matrix()

        # Calculates output for each layer and throw it as a input for the next one
        output = inputs
        for layer in range(0, self.layers_number):
            output = self.calc_layer_output(layer, output)

        # Returns output
        return output
