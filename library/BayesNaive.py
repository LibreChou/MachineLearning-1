import numpy as np
import math

class BayesNaive(object):

    def __init__(self, data, classes, continuous=False):
        # Sets properties
        self.data = data
        self.classes = classes
        self.continuous = continuous

        # Creates matrices if it's a continuous dataset
        if continuous:
            self.means = np.empty((0, 0), np.float64).tolist()
            self.standard = np.empty((0, 0), np.float64).tolist()

    # Gets the probability for one variable value
    @staticmethod
    def get_discrete_variable_probability(variable, value, data):
        samples = []
        for i in range(0, len(data)):
            if data[i][variable] == value:
                samples.append(data[i][variable])
        return float(len(samples)) / float(len(data))

    # Gets the probability of a class
    def get_discrete_class_probability(self, c, values):
        samples = []
        # Gets all samples fo this class
        for i in range(0, len(self.classes)):
            if self.classes[i] == c:
                samples.append(self.data[i])

        # Calculates classe's probability
        p_class = float(len(samples)) / float(len(self.data))

        # Calculates probability foreach variable value
        prob = 1.0
        for i in range(0, len(values)):
            prob *= self.get_discrete_variable_probability(i, values[i], samples)

        # Class probability * variables probabilities
        prob *= p_class
        return [c, prob]

    # Gets the most probable class given a value
    def get_discrete_probability(self, values):

        # Gets all classes values in the dataset
        unique_classes = []
        for c in self.classes:
            if c not in unique_classes:
                unique_classes.append(c)

        # Calculates probabilities foreach class
        probabilities = []
        for c in unique_classes:
            prob = self.get_discrete_class_probability(c, values)
            probabilities.append(prob)

        return probabilities

    def calc_classes_mean_and_variance(self, classes):
        # Matrices dimensions
        L = len(classes)

        # nxL matrices for mean and standard deviation
        self.means = np.empty((len(classes), len(self.data[0])), np.float64).tolist()
        self.standard = np.empty((len(classes), len(self.data[0])), np.float64).tolist()

        # For each class
        for c in range(0, L):

            # Gets all values given class c
            samples = []
            for i in range(0, len(self.classes)):
                if self.classes[i] == classes[c]:
                    samples.append(self.data[i])

            # Gets mean and variance for all values  in class c
            mean = np.array(samples).mean(axis=0).tolist()
            variance = np.array(samples).std(axis=0).tolist()

            # Store those values
            self.means[c] = mean
            self.standard[c] = variance

    # Uses a Gaussian to calculate the probability for class c given a value for a variable
    def calc_gaussian(self, c, variable, value):
        first_part = math.sqrt(2.0 * math.pi) * self.standard[c][variable]
        first_part = 1.0 / first_part

        second_part = (value - self.means[c][variable])**2 / (2.0 * (self.standard[c][variable])**2)
        second_part = (-1) * second_part
        second_part = math.exp(second_part)

        return first_part * second_part

    # Gets probability for continuous values given class c
    def get_continuous_class_probability(self, c, values, unique_classes):
        # Gets sample for class c
        samples = []
        for i in range(0, len(self.classes)):
            if self.classes[i] == unique_classes[c]:
                samples.append(self.data[i])

        # Gets class probability
        prob = float(len(samples)) / float(len(self.data))

        # Gets probability for each value given class C
        for i in range(0, len(values)):
            prob *= self.calc_gaussian(c, i, values[i])

        return [unique_classes[c], prob]

    # Gets probability for continuous values
    def get_continuous_probability(self, values):
        # Gets all classes values in the dataset
        unique_classes = []
        for c in self.classes:
            if c not in unique_classes:
                unique_classes.append(c)

        # Calculates the nxL mean and std deviation matrix
        self.calc_classes_mean_and_variance(unique_classes)

        # Calculates probabilities foreach class
        probabilities = []
        for c in range(0, len(unique_classes)):
            prob = self.get_continuous_class_probability(c, values, unique_classes)
            probabilities.append(prob)

        return probabilities

    def get_probability(self, values):
        # Selects if it will run the continuous or discrete learning algorithm
        if self.continuous:
            return self.get_continuous_probability(values)
        else:
            return self.get_discrete_probability(values)

    # Predicts the class given a value
    def predict(self, values):
        probabilities = self.get_probability(values)

        # Gets the total for all classes predictions
        total = 0.0
        for prob in probabilities:
            total += prob[1]

        # Applies Map Rule
        index = 0
        for i in range(1, len(probabilities)):
            if probabilities[i] > probabilities[index]:
                index = i

        # Returns class name and probability
        return [probabilities[index][0], probabilities[index][1] / total]
