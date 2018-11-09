from library.Neuron import *
import numpy as np

class NeuralNetwork(object):

    def __init__(self, layers_number):
        self.layers_number = layers_number
        self.layers = [layers_number]
        self.layers_mx = []

    def set_layer(self, layer, neurons):
        new_layer = []
        for neuron in neurons:
            new_layer.append(neuron)

        if self.layers[layer] is None:
            self.layers.append(new_layer)
        else:
            self.layers[layer] = new_layer

    def make_matrix(self):
        self.layers_mx = [self.layers_number]
        for i in range(0, self.layers_number):
            layer = []
            for neuron in self.layers[i]:
                layer.append(neuron.weights)
            self.layers_mx[i] = np.matrix(layer)

    def calc_layer_output(self, layer, inputs):
        inputs = np.matrix(inputs).transpose()
        layer_mx = self.layers_mx[layer]

        output = np.matmul(layer_mx, inputs)

        for i in range(0, output.shape[0]):
            value = self.layers[layer][i].fnet(output.item((i, 0)))
            output.itemset((i, 0), value)

        return output

    def process(self, inputs):
        if not self.layers_mx:
            self.make_matrix()
        output = inputs
        for layer in range(0, self.layers_number):
            output = self.calc_layer_output(layer, output)
        return output.tolist()[0]
