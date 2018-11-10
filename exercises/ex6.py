from library.NeuralNetwork import *
from library.Neuron import *
from library.Perceptron import *
from exercises.ex5 import get_iris_dataset
import json

def get_iris_dataset(path):
    # Fills inputs
    array = []
    classes = []

    # Reads and parse Json
    with open(path) as f:
        data = json.load(f)

    properties = ["sepalLength", "sepalWidth", "petalLength", "petalWidth"]
    for d in data:
        specie = d["species"]
        if specie == "setosa":
            c = 1
        elif specie == "versicolor":
            c = -1
        else:
            c = 0
        if c != 0:
            classes.append(c)
            array.append([float(d[p]) for p in properties])

    return array, classes


def test_perceptron(percep, test_data, title, print_res=True, labels=[]):
    print(title)
    count = len(labels) != 0
    n = len(test_data[0])
    sum = 0
    for data in range(0, len(test_data)):
        value = percep.process(test_data[data]).tolist()[0][0]
        if print_res:
            data_str = ""
            for v in range(0, n):
                if v < n - 1:
                    data_str += str(test_data[data][v]) + ", "
                else:
                    data_str += str(test_data[data][v])
            print("(" + data_str + "): " + str(value))
        if count and value == labels[data]:
            sum += 1
    if count:
        print("Acertou: " + str(sum) + " de " + str(len(labels)))

def ex6():
    # ==== Ex1 ====
    # Logic door dataset
    logic_dataset = [[0, 0], [0, 1], [1, 0], [1, 1]]

    # OR
    or_func = lambda x: 1 if x >= 1 else 0
    p_or = Perceptron(or_func, 2, 0.1, 0.1)
    p_or.layers[0][0].weights = [1, 1]
    or_labels = [0, 1, 1, 1]
    test_perceptron(p_or, logic_dataset, "==== OR ====", True, or_labels)

    # AND
    and_func = lambda x: 1 if x >= 2 else 0
    p_and = Perceptron(and_func, 2, 0.1, 0.1)
    p_and.layers[0][0].weights = [1, 1]
    and_labels = [0, 0, 0, 1]
    test_perceptron(p_and, logic_dataset, "==== AND ====", True, and_labels)

    # ==== Ex2: Run on Iris Dataset ====
    iris_dataset, iris_classes = get_iris_dataset("inputs/iris.json")
    iris_test, iris_test_classes = get_iris_dataset("inputs/iris_test.json")

    # Extends and shuffles the dataset items
    indexes = [random.randint(0, len(iris_dataset) - 1) for i in range(0, 200)]
    p_iris_data = [iris_dataset[index] for index in indexes]
    p_iris_classes = [iris_classes[index] for index in indexes]

    percep_func = lambda x: 1 if x >= 0 else -1
    rn_iris = Perceptron(percep_func, 4, 0.1, 0.1)
    rn_iris.train(p_iris_data, p_iris_classes)
    test_perceptron(rn_iris, iris_test, "==== Test on Iris Dataset =====", True, iris_test_classes)
