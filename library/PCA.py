import numpy as np

class PCA(object):

    # Initializes the object according to its inputs
    def __init__(self, inputs):

        # Initial checks
        if not isinstance(inputs, list) or len(inputs) == 0:
            raise ValueError("Bad inputs!")

        self.inputs = inputs

        # Gets the matrix's rows and columns number
        self.n = len(inputs)

        # Gets data length
        self.data_len = len(inputs[0])

        self.adjusted_values = []

        # Initializes matrix
        self.covariance_mx = self.get_covariance_mx()

        self.eigen_values, self.eigen_vectors = np.linalg.eig(self.covariance_mx)

        self.feature_vector = []


    def get_covariance_mx(self):

        # Adjust value for all variables
        for i in range(0, self.n):
            mean = np.mean(self.inputs[i])
            self.adjusted_values.append([])
            for j in range(0, self.data_len):
                self.adjusted_values[i].append(self.inputs[i][j] - mean)

        # Calcs the covariance matrix
        return np.cov(self.adjusted_values)

    # Chooses the eigen vector based on the precision needed
    def set_precision(self, percentage):
        self.feature_vector = []
        eigen_values = np.array(self.eigen_values)


        # Total sum of the eigenvalues
        sum = self.eigen_values.sum()

        precision = 0.0
        # Add EigenVecors until the desired precision is reached
        while precision < percentage:
            max_value = max(eigen_values)
            index = np.where(eigen_values == max_value)[0][0]
            values = []
            for j in range(0, self.eigen_vectors.shape[1]):
                values.append(self.eigen_vectors[j][index])
            self.feature_vector.append(values)
            precision += self.eigen_values.item(index) / sum
            eigen_values.itemset(index, 0.0)

        self.feature_vector = np.transpose(self.feature_vector)


    def get_final_data(self):
        feature_v = np.matrix(self.feature_vector).transpose()
        adjs = np.matrix(self.adjusted_values)
        return np.matmul(feature_v, adjs)

    def get_adjusted_data(self):
        feature_v = np.matrix(self.feature_vector).transpose()
        # Right inverse

        # Calculates pseudo inverse if not square
        if feature_v.shape[0] != feature_v.shape[1]:
            feature_v_inv = np.linalg.pinv(feature_v)
        else:
            feature_v_inv = np.linalg.inv(feature_v)

        output = np.matmul(feature_v_inv, np.matrix(self.get_final_data()))
        return output

    def get_original_data(self):
        data = self.get_adjusted_data()
        for i in range(0, len(self.inputs)):
            mean = np.mean(self.inputs[i])
            for value in data[i]:
                value += mean
        return data
