import numpy as np
from random import randint
import math


class Kmeans(object):

    def __init__(self, inputs, k):
        self.data = np.matrix(inputs).transpose().tolist()

        self.k = k
        self.n = len(self.data)

        # Get initial random
        self.center = []
        for i in range(0, k):
            index = randint(0, self.n - 1)
            while self.data[index] in self.center:
                index = randint(0, self.n - 1)
            self.center.append(self.data[index])


    @staticmethod
    def euclid_distance(p1, p2):
        return math.sqrt(sum([(a - b) ** 2 for a, b in zip(p1, p2)]))

    def get_groups(self, data):
        n_groups = len(self.center)
        n_data = len(data)
        distances = [999999999.0 for j in range(0, n_data)]
        groups = [0 for j in range(0, n_data)]

        for i in range(0, n_data):
            for j in range(0, n_groups):
                dist = self.euclid_distance(data[i], self.center[j])
                if dist < distances[i]:
                    distances[i] = dist
                    groups[i] = j
        return groups

    @staticmethod
    def matrix_equals(mx1, mx2):
        n_rows = len(mx1)
        n_columns = len(mx1[0])

        for i in range(0, n_rows):
            for j in range(0, n_columns):
                if mx1[i][j] != mx2[i][j]:
                    return False

        return True


    def move_centers(self, groups):
        new_centers = []
        for i in range(0, len(self.center)):
            items = []
            for j in range(0, len(self.data)):
                if groups[j] == i:
                    items.append(self.data[j])
            new_center = np.array(items).mean(axis=0).tolist()
            new_centers.append(new_center)
        if self.matrix_equals(new_centers, self.center):
            return False
        else:
            self.center = new_centers
            return True

    def execute(self):
        moved = True
        while moved:
            groups = self.get_groups(self.data)
            moved = self.move_centers(groups)
        return

