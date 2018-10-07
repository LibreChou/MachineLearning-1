from library.machine_learning import *
import re


def choose_lr_type():
    lrtype = int(input("Choose a type for the Linear Regression: \n1 - Linear\n2 - Quadratic\n3 - Weighted\nChoice: "))
    if lrtype == 1:
        lrtype = LRTypes.Linear
    elif lrtype == 2:
        lrtype = LRTypes.Quadratic
    elif lrtype == 3:
        lrtype = LRTypes.Weighted
    else:
        raise ValueError("Invalid linear regression type!")
    return lrtype


def ex1():
    # Arrange
    x = []
    y = []
    lrtype = 0
    valid_operation = True

    # Converts input arrays
    nx = int(input("Type the number of X elements (dimensions): "))


    # Inputs all dimensions
    for i in range(0, nx):
        inp = input("Please input the x" + str(i) + " axis with comma-separated values (e.g.: 1.0, 2.0, 3.0): ")
        x.append([])
        xstr = inp.split(',')
        for value in xstr:
            x[i].append(float(value))
    inp = input("Please input the y axis with comma-separated values (e.g.: 1.0, 2.0, 3.0): ")
    xstr = inp.split(',')
    for value in xstr:
        y.append(float(value))

    # Check for errors
    for values in x:
        if len(values) != len(y):
            raise ValueError("Bad Size inputs!")

    # Chooses the type of Linear Regression1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000
    lrtype = choose_lr_type()

    # Creates the Linear Regression context
    lr = LinearRegression(x, y, lrtype)

    # "Interactive" menu
    while valid_operation:
        opt = int(input("Choose and operation:\n1 - Solver for values:\n2 - Plot Graph\n3 - Change regression type\n other number = Exit\nChoice: "))
        if opt == 1:
            xstr = input("Input the X values in comma-separated form (e.g.: 1.0, 2.0, 3.0): ").split(',')
            values = []
            for s in xstr:
                values.append(float(s))
            print("Answer: " + lr.solve(values).__str__())
        elif opt == 2:
            print("bananas")
        elif opt == 3:
            lrtype = choose_lr_type()
            lr = LinearRegression(x, y, lrtype)
        else:
            print("See yah!!")
            valid_operation = False
        input("Press enter to continue...")