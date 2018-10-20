from enum import Enum
from library.LDA import *


class Iris_Types(Enum):
    setosa = 1
    versicolor = 2
    virginica = 3


def ex3():
    array = [[1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 3.0, 5.0, 6.0],
             [2.0, 3.0, 3.0, 5.0, 5.0, 0.0, 1.0, 1.0, 2.0, 3.0, 5.0]]
    classes = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]

    l1 = LDA(array, classes)
    print(np.matrix(l1.scatter_between))
    print(np.matrix(l1.scatter_within))
    #print(l1.classes)