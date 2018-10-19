from library.LDA import *

def ex3():
    array = [Sample([1.0, 2.0], Iris_Types.setosa), Sample([2.0, 3.0], Iris_Types.setosa),
             Sample([3.0, 3.0], Iris_Types.setosa), Sample([4.0, 5.0], Iris_Types.setosa),
             Sample([5.0, 5.0], Iris_Types.setosa),
             Sample([1.0, 0.0], Iris_Types.versicolor), Sample([2.0, 1.0], Iris_Types.versicolor),
             Sample([3.0, 1.0], Iris_Types.versicolor), Sample([3.0, 2.0], Iris_Types.versicolor),
             Sample([5.0, 3.0], Iris_Types.versicolor), Sample([6.0, 5.0], Iris_Types.versicolor)]

    l1 = LDA(array)
    l1.get_lines()
    #print(l1.classes)