from exercises.exercises import *
from exercises.ex2 import *


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt



def main():
     ex = int(input("Type the exercise number as an integer (e.g.: 1): "))

     if (ex == 1):
          ex1()
     elif(ex == 2):
          ex2()


main()
