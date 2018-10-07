from enum import Enum
from library.math import *
from math import *
import numpy
import matplotlib.pyplot as plt


class LRTypes(Enum):
    Linear = 1
    Quadratic = 2
    Weighted = 3


class LinearRegression(object):

    def __init__(self, x, y, lrtype):
        # Initialize properties
        self.betas = TeaMatrix(numpy.empty((0, 0)))
        self.x = numpy.matrix(x, numpy.float64)
        self.y = numpy.array(y, numpy.float64)
        # Initializes with "no weights" (weights == 1.0)
        self.weights = numpy.empty(len(self.y), numpy.float64)
        for i in range(0, len(self.weights)):
            self.weights[i] = 1.0
        self.type = lrtype

        # Initial checks
        if self.x.shape[1] != len(y) or self.x.shape[1] == 0 or len(y) == 0:
            raise ValueError("Bad Size inputs!")

        # Creates the linear regression context
        self.linear()

    def format_input(self):
        # Arrange
        x = numpy.empty((0, 0), numpy.float64)
        y = numpy.empty((0, 0), numpy.float64)
        n = self.x.shape[0]
        m = self.x.shape[1]

        # Formats input according to regression type
        # Linear Regression -> [1, x1, x2...]
        if self.type == LRTypes.Linear or self.type == LRTypes.Weighted:
            x = numpy.empty((m, n + 1), numpy.float64)
            y = numpy.empty((len(self.y), 1), numpy.float64)
            for i in range(0, m):
                for j in range(0, n + 1):
                    if j == 0:
                        x.itemset((i, j), 1.0)
                    else:
                        x.itemset((i, j), self.x.item((j - 1, i)))
            for i in range(0, m):
                y.itemset((i, 0), self.y.item(i))
        # Quadratic Linear Regression -> [1, x, x^2]
        elif self.type == LRTypes.Quadratic:
            x = numpy.empty((m, 3), numpy.float64)
            y = numpy.empty((len(self.y), 1), numpy.float64)
            for i in range(0, m):
                x.itemset((i, 0), 1.0)
                x.itemset((i, 1), self.x.item(i))
                x.itemset((i, 2), pow(self.x.item(i), 2))
            for i in range(0, m):
                y.itemset((i, 0), self.y.item(i))
        return x, y

    def format_solve_input(self, x):
        # Arranges
        input_mx = None

        # Formats input to solve system according to type
        # Linear Regression -> [1, x...]
        if self.type == LRTypes.Linear or self.type == LRTypes.Weighted:
            input_mx = numpy.empty((1, len(x) + 1), numpy.float64)
            for i in range(0, len(x) + 1):
                if i == 0:
                    input_mx.itemset((0, 0), 1.0)
                else:
                    input_mx.itemset((0, i), x[i - 1])

        # Quadratic Linear Regression -> [1, x, x^2]
        elif self.type == LRTypes.Quadratic:
            input_mx = numpy.empty((1, 3), numpy.float64)
            input_mx.itemset((0, 0), 1.0)
            input_mx.itemset((0, 1), x[0])
            input_mx.itemset((0, 2), pow(x[0], 2))

        return input_mx

    def calcBetas(self):
        # Gets formatted input
        x, y = self.format_input()

        mx = TeaMatrix(x)
        my = TeaMatrix(y)
        mw = TeaMatrix(numpy.empty((len(self.weights), len(self.weights)), numpy.float64))

        # Creates the W matrix to multiply the weights
        for i in range(0, len(self.weights)):
            for j in range(0, len(self.weights)):
                mw.matrix.itemset((i, j), 0.0)
            mw.matrix.itemset((i, i), self.weights.item(i))

        # Transposes X
        mxt = mx.transpose()

        # Calculates X^2 with weights
        mxtw = mxt.multiply(mw)

        # Calculates XTW * X
        mx2 = mxtw.multiply(mx)

        # Multiples weighted transposed X by Y
        mxty = mxtw.multiply(my)

        # Inverts X^2
        mx2i = mx2.invert()

        # Stores Results in the betas matrix
        self.betas = mx2i.multiply(mxty)

    def linear(self):
        self.calcBetas()

        # Calculates the new weights
        if self.type == LRTypes.Weighted:
            for i in range(0, len(self.y)):
                current = self.y.item(i)
                x = []
                # builds the X vector to the solution
                for j in range(0, self.x.shape[0]):
                    x.append(self.x.item(j, i))
                value = 1.0 / abs(current - self.solve(x))
                self.weights.itemset(i, value)
            # Re-calculates new betas for the new weights
            self.calcBetas()

    def solve(self, x):
        # Arrange
        if not isinstance(x, (list, numpy.ndarray)):
            x = [x]
        input_mx = self.format_solve_input(x)

        # Initial checks
        if input_mx.shape[1] != self.betas.get_rows:
            raise ValueError("Bad argument size!!")

        # Mathematics operation
        mx = TeaMatrix(input_mx)
        res = mx.multiply(self.betas)

        # The matrix's first item is the result
        return res.matrix.item((0, 0))

    def plot(self):
        mx = TeaMatrix(self.x)
        mxt = mx.transpose()
        yres = []
        for i in range(0, mxt.get_rows):
            yres.append(self.solve(numpy.array(mxt.matrix[i])[0]))
        try:
            plt.style.use('seaborn-whitegrid')
            plt.plot(self.x, numpy.matrix(self.y), 'ro')
            plt.plot(self.x, numpy.matrix(yres), 'bo')

            plt.show()
        except ValueError:
            print("This module can only plot for 2 dimensional arrays!")
        finally:
            print("The func output will be plotted bellow:")
            print(yres)