import numpy


class TeaMatrix(object):

    @property
    def get_rows(self):
        return self.matrix.shape[0]

    @property
    def get_columns(self):
        return self.matrix.shape[1]

    @property
    def is_square(self):
        return self.get_rows == self.get_columns

    def __init__(self, data):
        self.matrix = numpy.matrix(data, numpy.float64)
        self.index = numpy.empty((0, 0))
        self.pivotM = numpy.empty((0, 0), numpy.float64)
        self.index = numpy.empty(0)
        self.detParity = 1.0

    def __str__(self):
        return self.matrix.__str__()

    def multiply(self, other):
        return TeaMatrix(numpy.matmul(self.matrix, other.matrix))

    def transpose(self):
        return TeaMatrix(self.matrix.transpose())

    # Generates and set the resultant matrix of the partial pivoting
    def set_pivot_matrix(self):
        # Raises error if it's a square matrix
        if not self.is_square:
            raise ValueError("Bad matrix size!")

        # Arrange variables
        self.pivotM = numpy.matrix(self.matrix)
        n = self.get_rows
        tiny = 1.0e-40
        self.detParity = 1.0
        self.index = numpy.empty(n, numpy.int)
        scales = numpy.empty(n, numpy.float64)

        # Gets the scale for each row (the biggest element)
        for i in range(0, n):
            big = 0.0
            for j in range(0, n):
                temp = abs(self.pivotM.item(i, j))
                if temp > big:
                    big = temp
            if big == 0.0:
                raise ValueError("Bad operation with a singular Matrix!")
            scales[i] = 1.0 / big

        # Looks for the largest pivot element
        for k in range(0, n):
            big = 0.0

            # Finds the biggest pivot
            for i in range(k, n):
                temp = scales.item(i) * abs(self.pivotM.item((i, k)))
                if temp > big:
                    big = temp
                    imax = i

            # Exchange rows if needed
            if k != imax:
                for j in range(0, n):
                    temp = self.pivotM.item((imax, j))
                    self.pivotM.itemset((imax, j), self.pivotM.item((k, j)))
                    self.pivotM.itemset((k, j), temp)
                self.detParity = - self.detParity
                scales.itemset(imax, scales.item(k))

            self.index[k] = imax
            if self.pivotM.item((k, k)) == 0.0:
                self.pivotM.itemset((k, k), tiny)

            # Divide by the pivot
            for i in range(k + 1, n):
                value = self.pivotM.item((i, k)) / self.pivotM.item((k, k))
                self.pivotM.itemset((i, k), value)
                temp = self.pivotM.item((i, k))
                for j in range(k + 1, n):
                    value = self.pivotM.item((i, j)) - temp * self.pivotM.item((k, j))
                    self.pivotM.itemset((i, j), value)

    def solve_column_for(self, b):
        # Initial checks to avoid erros
        if not self.is_square:
            raise ValueError("Cannot solve for a non-square matrix!!")
        n = self.get_columns
        if len(b) != n:
            raise ValueError("Bad size! System cannot be solved!")

        # Arrange
        sum = 0.0
        ii = 0
        output = numpy.array(b, numpy.float64)

        # Ensures the pivot matrix is calculated
        if len(self.pivotM) == 0:
            self.set_pivot_matrix()

        # Applies formula 2.3.6
        for i in range(0, n):
            ip = self.index[i]
            sum = output[ip]
            output[ip] = output[i]
            if ii != 0:
                for j in range(ii - 1, i):
                    sum -= self.pivotM.item((i, j)) * output[j]
            elif sum != 0.0:
                ii = i + 1
            output[i] = sum

        # Applies formula 2.3.7
        for i in range(n - 1, -1, -1):
            sum = output[i]
            for j in range(i + 1, n):
                sum -= self.pivotM.item((i, j)) * output[j]
            output[i] = sum / self.pivotM.item((i, i))

        return output

    def solve_for(self, other):
        # Initial checks
        if not self.is_square:
            raise ValueError("Cannot solve for a non-square matrix!!")
        n = self.get_columns
        ncolumns = len(other)
        if ncolumns != n:
            raise ValueError("Cannot solve: bad size error!")

        b = numpy.matrix(other, numpy.float64)

        # Arrange
        output = numpy.empty((n, n))
        column = numpy.empty(n)

        # Ensures the pivot matrix is calculated
        if len(self.pivotM) == 0:
            self.set_pivot_matrix()

        # Iterates through all columns and solve them
        for j in range(0, ncolumns):
            for i in range(0, n):
                column.itemset(i, b.item((i, j)))
            res = self.solve_column_for(column)
            for i in range(0, n):
                output.itemset((i, j), res[i])
        return TeaMatrix(output)

    def invert(self):
        # Initial check
        if not self.is_square:
            raise ValueError("Cannot solve for a non-square matrix!!")

        # Arrange
        n = self.get_columns
        identity = numpy.empty((n, n), numpy.float64)
        # Creates the identity matrix
        for i in range(0, n):
            for j in range(0, n):
                identity.itemset((i, j), 0.0)
            identity.itemset((i, i), 1.0)

        # Solve the identity matrix and return it
        return self.solve_for(identity)