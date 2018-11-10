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

    for d in data:
        if d["species"] == "setosa" or d["species"] == "versicolor":
            classes.append(1) if d["species"] == "setosa" else classes.append(-1)
            new_data = []
            new_data.append(float(d["sepalLength"]))
            new_data.append(float(d["sepalWidth"]))
            new_data.append(float(d["petalLength"]))
            new_data.append(float(d["petalWidth"]))
            array.append(new_data)

    return array, classes

def test_logic_port(rn, test_data, title, print_res=True, labels = []):
    print(title)
    sum = 0
    for data in range(0, len(test_data)):
        value = rn.process(test_data[data])[0]
        if print_res:
            print("(" + str(test_data[data][0]) + "," + str(test_data[data][1]) + "): " + str(value))
        if len(labels) != 0 and value == labels[data]:
            sum += 1
    if len(labels) != 0:
        print("Acertou: " + str(sum) + " de " + str(len(labels)))

def ex6():
    # Test
    activation_func = lambda x: 1 if x >= 0 else -1
    neurons = [
            Neuron([3, 1, 1, -3, 1], activation_func),
            Neuron([-1, 1, 1, 1, -3], activation_func),
            Neuron([-1, -3, 1, 1, 1], activation_func),
            Neuron([-1, 1, -3, 1, 1], activation_func)
        ]
    rn = NeuralNetwork(1)
    rn.set_layer(0, neurons)
    rn.make_matrix()
    res = rn.calc_layer_output(0, [1, 1, 1, -1, -1])
    print(res)

    # Ex 1
    test_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
    test_labels = [-1, -1, -1, 1]
    test_labels_or = [-1, 1, 1, 1]

    or_func = lambda x: 1 if x >= 2 else 0
    neurons_or = [Neuron([1, 1], or_func)]
    rn_or = NeuralNetwork(1)
    rn_or.set_layer(0, neurons_or)

    test_logic_port(rn_or, test_data, "==== Porta Lógica AND ====")

    and_func = lambda x: 1 if x >= 1 else 0
    neurons_and = [Neuron([1, 1], and_func)]
    rn_and = NeuralNetwork(1)
    rn_and.set_layer(0, neurons_and)

    test_logic_port(rn_and, test_data, "==== Porta Lógica OR ====")

    # Ex2

    # test 2
    func_perceptron = lambda x: 1 if x >= 0 else -1

    train_data = []
    train_labels = []
    for i in range(0, 200):
        index = random.randint(0, 3)
        train_data.append(test_data[index])
        train_labels.append(test_labels[index])

    rn_test = Perceptron(func_perceptron, 2, 0.1, 0.10)
    rn_test.train(train_data, train_labels)

    # Run on Iris Dataset
    iris_dataset, iris_classes = get_iris_dataset("inputs/iris.json")
    iris_test, iris_test_classes = get_iris_dataset("inputs/iris_test.json")

    p_iris_data = []
    p_iris_classes = []

    for i in range(0, len(iris_dataset) * 10):
        index = random.randint(0, len(iris_dataset) - 1)
        p_iris_data.append(iris_dataset[index])
        p_iris_classes.append(iris_classes[index])

    rn_iris = Perceptron(func_perceptron, 4, 0.1, 0.1)
    rn_iris.train(p_iris_data, p_iris_classes)
    test_logic_port(rn_iris, iris_test, "==== Teste iris =====", False, iris_test_classes)
