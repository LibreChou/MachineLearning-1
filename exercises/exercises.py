from library.machine_learning import *
import re


def choose_lr_type(inp_size):
    lrtype = 0
    if inp_size == 1 or inp_size == 2:
        lrtype = int(input("Choose a type for the Linear Regression: \n1 - Linear\n2 - Weighted\n3 - Quadratic\nChoice: "))
    else:
        lrtype = int(input("Choose a type for the Linear Regression: \n1 - Linear\n2 - Weighted\nChoice: "))

    # Sets the Linear Regression type enum
    if lrtype == 1:
        lrtype = LRTypes.Linear
    elif lrtype == 2:
        lrtype = LRTypes.Weighted
    elif lrtype == 3:
        lrtype = LRTypes.Quadratic
    else:
        raise ValueError("Invalid linear regression type!")
    return lrtype

def ex1_get_inputs():
    # Arrange
    x = []
    y = []

    # Converts input arrays
    nx = int(input("Type the number of X axis (dimensions in X): "))

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

    # Case its square and 3 dimensions
    if inp_size == 2 or LinearRegression == LRTypes.Quadratic:
        x.append([])
        for i in range(0, len(x[0])):
            x[2].append(pow(x[0][i], 2))
        x.append([])
        for i in range(0, len(x[0])):
            x[3].append(pow(x[1][i], 2))
        x.append([])
        for i in range(0, len(x[0])):
            x[4].append(2 * x[0][i] * x[1][i])
        # Creates the Linear Regression context
        lr = LinearRegression(x, y, lrtype, True)
    else:
        # Creates the Linear Regression context
        lr = LinearRegression(x, y, lrtype)

    # "Interactive" menu
    while valid_operation:
        opt = int(input("Choose and operation:\n1 - Solver for values\n2 - Plot Graph\n3 - Change regression type\n other number = Exit\nChoice: "))
        if opt == 1:
            xstr = input("Input the X values in comma-separated form (e.g.: 1.0, 2.0, 3.0): ").split(',')
            values = []
            resp = ""
            if inp_size == 2 and lrtype == LRTypes.Quadratic:
                values = [float(xstr[0]), float(xstr[1]), pow(float(xstr[0]), 2), pow(float(xstr[1]), 2), 2.0 * float(xstr[0])* float(xstr[1])]
                resp = lr.solve(values, True).__str__()
            else:
                for s in xstr:
                    values.append(float(s))
                resp = lr.solve(values).__str__()
            print("Answer: " + resp)
        elif opt == 2:
            if inp_size == 2 and lrtype == LRTypes.Quadratic:
                lr.plot(True)
            else:
                lr.plot()
        elif opt == 3:
            lrtype = choose_lr_type(inp_size)
            force = False
            if inp_size == 2 and len(x) == 2 and lrtype == LRTypes.Quadratic:
                x.append([])
                for i in range(0, len(x[0])):
                    x[2].append(pow(x[0][i], 2))
                x.append([])
                for i in range(0, len(x[0])):
                    x[3].append(pow(x[1][i], 2))
                x.append([])
                for i in range(0, len(x[0])):
                    x[4].append(2 * x[0][i] * x[1][i])
                force = True
            elif inp_size == 2 and len(x) == 5 and lrtype != LRTypes.Quadratic:
                x.pop(4)
                x.pop(3)
                x.pop(2)
            lr = LinearRegression(x, y, lrtype, force)
        else:
            print("See yah!!")
            valid_operation = False
        input("Press enter to continue...")