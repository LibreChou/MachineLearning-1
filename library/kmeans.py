import numpy as np
from random import randint
import math


class Kmeans(object):

    def __init__(self, inputs, k):
        self.data = np.matrix(inputs).transpose().tolist()

        # Set properties
        self.k = k
        self.n = len(self.data)
        self.center = []

    # Get initial random centers
    def init_centers(self):
        self.center = []
        for i in range(0, self.k):
            index = randint(0, self.n - 1)
            # Make sure it is not a repeated center
            while self.data[index] in self.center:
                index = randint(0, self.n - 1)
            self.center.append(self.data[index])

    # Calculates Euclidian distance
    @staticmethod
    def euclid_distance(p1, p2):
        return math.sqrt(sum([(a - b) ** 2 for a, b in zip(p1, p2)]))

    # Defines which group each point belongs to
    def get_groups(self, data):
        n_groups = len(self.center)
        n_data = len(data)
        distances = [999999999.0 for j in range(0, n_data)]
        groups = [0 for j in range(0, n_data)]

        # For all data
        for i in range(0, n_data):
            # Calc the distance between data point to all center
            for j in range(0, n_groups):
                dist = self.euclid_distance(data[i], self.center[j])
                # The lowest distance wins and the group's index is added
                if dist < distances[i]:
                    distances[i] = dist
                    groups[i] = j
        # return group indexes
        return groups

    # Checks if two matrices are equals
    @staticmethod
    def matrix_equals(mx1, mx2):
        n_rows = len(mx1)
        n_columns = len(mx1[0])

        for i in range(0, n_rows):
            for j in range(0, n_columns):
                if mx1[i][j] != mx2[i][j]:
                    return False

        return True

    # Move centers to new point according to the groups' means
    def move_centers(self, groups):
        new_centers = []

        # For all centers
        for i in range(0, len(self.center)):

            # Gets group related to it
            items = []
            for j in range(0, len(self.data)):
                if groups[j] == i:
                    items.append(self.data[j])

            # New Center is the group's mean
            new_center = np.array(items).mean(axis=0).tolist()
            new_centers.append(new_center)

        # Returns False if centers not changed or true otherwise
        if self.matrix_equals(new_centers, self.center):
            return False
        else:
            self.center = new_centers
            return True

    # Executes K-means
    def execute(self):
        # Initializes random centers
        self.init_centers()
        moved = True
        while moved:
            # Gets the group each point belongs to (nearest center)
            groups = self.get_groups(self.data)
            # Move to new groups' centers
            moved = self.move_centers(groups)

