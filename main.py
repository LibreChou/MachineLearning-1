from exercises.exercises import *
from exercises.ex2 import *
from exercises.ex3 import *
from exercises.ex4 import *
from exercises.ex5 import *


def main():
     ex = int(input("Type the exercise number as an integer (e.g.: 1): "))
     # ex = 5
     if ex == 1:
          ex1()
     elif ex == 2:
          ex2()
     elif ex == 3:
          ex3()
     elif ex == 4:
          ex4()
     elif ex == 5:
          ex5()


main()
