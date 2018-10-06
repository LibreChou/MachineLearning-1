from library import math


def main():
    inp = [69, 67, 71, 65, 72, 68, 74, 65, 66, 72]
    y = [9.5, 8.5, 11.5, 10.5, 11, 7.5, 12, 7, 7.5, 13]
    mx_inp = []
    for number in inp:
        mx_inp.append([1, number])
    print(mx_inp)
    mx = math.TeaMatrix(mx_inp)
    mxt = mx.transpose()
    my = math.TeaMatrix(y).transpose()

    mx2 = mxt.multiply(mx)

    print("======")


    # print(mx2)
    # print(mxT.matrix)
    # xty = mxt.multiply(my)
    # print(xty)

    # print(mx2)
    # mx2.lu_decomposition()

    # mx2.invert()

def main2():
    teste = math.TeaMatrix([[10, 689], [689, 47565]])
    print("=======")
    testeinv = teste.invert()
    # print(teste)
    print(testeinv)
    mul = math.TeaMatrix([[98], [6800]])
    res = testeinv.multiply(mul)
    print(res)
    # print(teste.multiply(testeinv))
main2()