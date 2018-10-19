import numpy as np
from enum import Enum
import matplotlib.pyplot as plt

class Iris_Types(Enum):
    setosa = 1
    versicolor = 2
    virginica = 3


# A class that represents a sample with dimensions and classification
class Sample(object):
    def __init__(self, variables, classification):
        # Initial checks
        if len(variables) == 0:
            raise ValueError("Cannot initialize with empty inputs!")

        # Set Properties
        self.variable = variables
        self.classification = classification


class LDA(object):
    def __init__(self, samples):
        # Initial checks
        if len(samples) == 0:
            raise ValueError("Cannot initialize with empty inputs!")

        # Defining Properties
        self.input = samples

        # Defines all classes
        self.classes = []
        for sample in samples:
            if not sample.classification in self.classes:
                self.classes.append(sample.classification)

        # Scatter between
        self.scatter_b = self.get_scatter_between(samples)

        # Scatter within
        self.scatter_w = self.get_scatter_within(samples)

    def get_variable_mean(self, inputs, i):
        mean = 0.0
        for j in range(0, len(inputs)):
            mean += inputs[j].variable[i]
        mean = mean / len(inputs)
        return mean

    def get_center_of_data(self, inputs):
        # Initial checks
        if len(inputs) == 0:
            raise ValueError("Cannot operate with empty inputs!")

        n_dimensions = len(inputs[0].variable)
        result = np.empty((1, n_dimensions), np.float64)

        for i in range(0, n_dimensions):
            mean = self.get_variable_mean(inputs, i)
            result.itemset((0, i), mean)
        return result

    def get_scatter_between(self, inputs):

        # Divide by classes
        groups = []
        for group in self.classes:
            values = []
            for data in inputs:
                if data.classification == group:
                    values.append(data)
            groups.append(tuple((group, values)))

        # Get the center
        center = self.get_center_of_data(inputs)

        # Initializes matrix
        sb = np.empty((center.shape[1], center.shape[1]), np.float64)
        for i in range(0, sb.shape[0]):
            for j in range(0, sb.shape[1]):
                sb.itemset((i, j), 0.0)

        # Number of samples
        n = len(inputs)

        # Calculates scatter between
        for group in groups:

            group_mean = []
            # calc mean foreach variable
            for i in range(0, center.shape[1]):
                mean_value = self.get_variable_mean(group[1], i)
                group_mean.append(mean_value)
            ni = len(group[1])
            diff = np.subtract(group_mean, center)
            diff_t = diff.transpose()
            mul = np.multiply(diff, diff_t)
            final = np.multiply(ni, mul)
            sb = np.add(sb, final)

        print(sb)
        return sb

    def get_scatter_within(self, inputs):

        # Divide by classes
        groups = []
        for group in self.classes:
            values = []
            for data in inputs:
                if data.classification == group:
                    values.append(data)
            groups.append(tuple((group, values)))


        # Initializes matrix
        n = len(groups[0][1][0].variable)
        sw = np.empty((n, n), np.float64)
        for i in range(0, sw.shape[0]):
            for j in range(0, sw.shape[1]):
                sw.itemset((i, j), 0.0)

        # Finds Scatter Within
        for group in groups:
            ni = len(group[1])

            group_mean = []
            # calc mean foreach variable
            for i in range(0, n):
                mean_value = self.get_variable_mean(group[1], i)
                group_mean.append(mean_value)

            # Subtracts values from group mean
            for i in range(0, ni):
                diff = np.subtract(np.matrix(group[1][i].variable), np.matrix(group_mean))
                diff_t = diff.transpose()
                mul = np.multiply(diff, diff_t)
                sw = np.add(sw, mul)

        print(sw)
        return sw

    def get_lines(self):
        sw_inv = np.linalg.inv(self.scatter_w)
        mx = np.multiply(sw_inv, self.scatter_b)

        eigen_values, eigen_vectors = np.linalg.eig(mx)
        print(eigen_values)
        print(eigen_vectors)
        data = []

        for i in range(0, len(self.input[0].variable)):
            data.append([])
            for j in range(0, len(self.input)):
                data[i].append(self.input[j].variable[i])



        feature = np.matrix([-0.18051168, -0.98357284])

        featurev2 = np.matrix([-0.98357284, 0.18051168])



        data_mx = np.matrix(data)
        finaldata = np.matmul(feature, data_mx)

        finaldata2 = np.matmul(featurev2, data_mx)

        originaldata = np.matmul(np.linalg.pinv(feature), finaldata)
        originaldata2 = np.matmul(np.linalg.pinv(featurev2), finaldata2)
        d1 = [0.38767724, 0.59780809, 0.63039256, 1.0180698,  1.05065426, 0.03258447, 0.24271532, 0.27529979, 0.45284617, 0.69556149, 1.08323873]
        d2 = [2.11237745, 3.25733937, 3.43488576, 5.54726321, 5.7248096,  0.17754639, 1.32250831, 1.50005469, 2.46747022, 3.78997853, 5.90235598]
        #for i in range(0, len(self.input[0].variables)):
        plt.scatter(data[0], data[1])
        plt.plot(originaldata.tolist()[0], originaldata.tolist()[1])
        plt.plot(originaldata2.tolist()[0], originaldata2.tolist()[1])

        return eigen_vectors
