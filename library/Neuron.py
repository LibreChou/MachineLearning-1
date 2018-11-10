

# This class represents the basic properties of a generic neuron
class Neuron(object):

    def __init__(self, weights, fnet):
        # Array of weights
        self.weights = weights
        # Lambda function representing the neuron function
        self.fnet = fnet