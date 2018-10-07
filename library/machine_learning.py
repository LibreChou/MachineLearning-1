from enum import Enum
from library.math import *
from math import *
import numpy


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
        self.weights = numpy.empty(len(self.y), numpy.float64)
        self.type = lrtype

        # Initial checks
        if self.x.shape[1] != len(y) or self.x.shape[1] == 0 or len(y) == 0:
            raise ValueError("Bad Size inputs!")

        # Create the Linear Regression according with the type specified
        if lrtype == LRTypes.Linear:
            print("Hey")
        self.linear()

    def format_input(self):
        # Arrange
        x = numpy.empty((0, 0), numpy.float64)
        y = numpy.empty((0, 0), numpy.float64)
        n = self.x.shape[0]
        m = self.x.shape[1]

        # Formats input according to regression type
        # Linear Regression -> [1, x]
        if self.type == LRTypes.Linear:
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
        if self.type == LRTypes.Linear:
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

    def linear(self):
        # Gets formatted input
        x, y = self.format_input()

        mx = TeaMatrix(x)
        my = TeaMatrix(y)

        # Transposes X
        mxt = mx.transpose()

        # Calculates X^2
        mx2 = mxt.multiply(mx)

        # Multiples transposed X by Y
        mxty = mxt.multiply(my)

        # Inverts X^2
        mx2i = mx2.invert()

        # Stores Results in the betas matrix
        self.betas = mx2i.multiply(mxty)

        if self.type == LRTypes.Weighted:
            w = TeaMatrix(numpy.empty(()))

    def solve(self, x):
        # Arrange
        input_mx = self.format_solve_input(x)

        # Initial checks
        if input_mx.shape[1] != self.betas.get_rows:
            raise ValueError("Bad argument size!!")

        # Mathematics operation
        mx = TeaMatrix(input_mx)
        res = mx.multiply(self.betas)

        # The matrix's first item is the result
        return res.matrix.item((0, 0))
