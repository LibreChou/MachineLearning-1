from library.machine_learning import *
import re


def choose_lr_type(inp_size):
    lrtype = 0
    if inp_size == 1:
        lrtype = int(input("Choose a type for the Linear Regression: \n1 - Linear\n2 - Weighted\n3 - Quadratic\nChoice: "))
    else:
        lrtype = int(input("Choose a type for the Linear Regression: \n1 - Linear\n2 - Weighted\nChoice: "))

    # Sets the Linear Regression type enum
    if lrtype == 1:
        lrtype = LRTypes.Linear
    elif lrtype == 2:
        lrtype = LRTypes.Weighted
    elif lrtype == 3 and inp_size == 1:
        lrtype = LRTypes.Quadratic
    else:
        raise ValueError("Invalid linear regression type!")
    return lrtype

def ex1_get_inputs():
    # Arrange
    x = []
    y = []

    # Converts input arrays
    nx = int(input("Type the number of X elements (dimensions): "))

    # Checks if size if valid
    if nx < 1:
        raise ValueError("Bad value for input size!!")

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
    return x, y, nx

def ex1():

    lrtype = 0
    valid_operation = True

    # Gets inputs and dimension
    x, y, inp_size = ex1_get_inputs()

    # Chooses the type of Linear Regression
    lrtype = choose_lr_type(inp_size)

    # Creates the Linear Regression context
    lr = LinearRegression(x, y, lrtype)

    # "Interactive" menu
    while valid_operation:
        opt = int(input("Choose and operation:\n1 - Solver for values\n2 - Plot Graph\n3 - Change regression type\n other number = Exit\nChoice: "))
        if opt == 1:
            xstr = input("Input the X values in comma-separated form (e.g.: 1.0, 2.0, 3.0): ").split(',')
            values = []
            for s in xstr:
                values.append(float(s))
            print("Answer: " + lr.solve(values).__str__())
        elif opt == 2:
            lr.plot()
        elif opt == 3:
            lrtype = choose_lr_type(inp_size)
            lr = LinearRegression(x, y, lrtype)
        else:
            print("See yah!!")
            valid_operation = False
        input("Press enter to continue...")