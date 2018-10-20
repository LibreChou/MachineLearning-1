import numpy as np
import matplotlib.pyplot as plt



class LDA(object):
    def __init__(self, samples, classes):
        # Initial checks
        if len(samples) == 0 or len(classes) == 0:
            raise ValueError("Cannot initialize with empty inputs!")

        # Initialize properties
        self.samples = samples
        self.classes = classes
        self.dimensions = len(samples)
        self.n = len(classes)

        # Sets groups amount
        self.unique_classes = []
        for i in range(0, self.n):
            if not self.classes[i] in self.unique_classes:
                self.unique_classes.append(self.classes[i])

        # Gets central point
        self.mean = []
        for i in range(0, self.dimensions):
            self.mean.append(np.mean(samples[i]))

        self.scatter_between = self.scatter_b()
        self.scatter_within = self.scatter_w()

    # Creates 0.0 filled matrix
    @staticmethod
    def empty_mx(rows, columns):
        res = np.empty((rows, columns), np.float64)
        for i in range(0, res.shape[0]):
            for j in range(0, res.shape[1]):
                res.itemset((i, j), 0.0)
        return res

    def scatter_b(self):
        sb = self.empty_mx(self.dimensions, self.dimensions)

        t_samples = np.matrix(self.samples).transpose().tolist()

        # Applies scatter between matrix formula
        for c in self.unique_classes:
            elements = []
            for i in range(0, self.n):
                if self.classes[i] == c:
                    elements.append(t_samples[i])

            samples = np.matrix(elements).transpose().tolist()
            # Gets the mean for the class c
            class_mean = []
            for i in range(0, self.dimensions):
                class_mean.append(np.mean(samples[i]))

            # Applies formula
            diff = np.matrix(np.subtract(class_mean, self.mean))
            diff_transposed = diff.transpose()
            mul = np.matmul(diff_transposed, diff)
            value = np.multiply(len(elements), mul)
            sb = np.add(sb, value)

        return sb.tolist()

    def scatter_w(self):
        sw = self.empty_mx(self.dimensions, self.dimensions)

        t_samples = np.matrix(self.samples).transpose().tolist()

        # Applies scatter withing formula
        for c in self.unique_classes:
            elements = []
            for i in range(0, self.n):
                if self.classes[i] == c:
                    elements.append(t_samples[i])

            samples = np.matrix(elements).transpose().tolist()
            # Gets the mean for the class c
            class_mean = []
            for i in range(0, self.dimensions):
                class_mean.append(np.mean(samples[i]))

            # Applies formula
            for i in range(0, len(elements)):
                diff = np.matrix(np.subtract(elements[i], class_mean))
                diff_transposed = diff.transpose()
                mul = np.matmul(diff_transposed, diff)
                sw = np.add(sw, mul)

        return sw.tolist()

    def plot_separation(self):
        # Gets eigensystem to mx
        sw_inv = np.matrix(np.linalg.pinv(self.scatter_within))
        mx = np.matmul(sw_inv, np.matrix(self.scatter_between))

        eigen_values, eigen_vectors = np.linalg.eig(mx)

        print(eigen_values)
        print(eigen_vectors)



        # It only plots if dimension is 2
        if self.dimensions == 2:
            t_samples = np.matrix(self.samples).transpose().tolist()

            # Prints scattered points
            for c in self.unique_classes:
                elements = []
                for i in range(0, self.n):
                    if self.classes[i] == c:
                        elements.append(t_samples[i])
                elements = np.matrix(elements).transpose().tolist()
                plt.scatter(elements[0], elements[1])

            # Prints eigenvectors
            eigen_t = eigen_vectors.transpose().tolist()
            for i in range(0, np.matrix(eigen_t).shape[0]):
                if eigen_values[i] != 0.0:
                    data = np.matmul(np.matrix(eigen_t[i]), np.matrix(self.samples))
                    eigen_inv = np.linalg.pinv(np.matrix(eigen_t[i]))
                    finaldata = np.matmul(eigen_inv, data).tolist()
                    for j in range(0, self.n):
                        finaldata[0][j] += self.mean[0]
                        finaldata[1][j] += self.mean[1]
                    plt.plot(finaldata[0], finaldata[1], "r-")

                    # Prints scattered points after eigen
                    t_final = np.linalg.inv(np.matrix(finaldata))
                    for c in self.unique_classes:
                        elements = []
                        for i in range(0, self.n):
                            if self.classes[i] == c:
                                elements.append(t_samples[i])
                        elements = np.matrix(elements).transpose().tolist()
                        plt.scatter(elements[0], elements[1])


        plt.show()