from library.NeuralNetwork import *
from library.Neuron import *
from library.Perceptron import *
from exercises.ex5 import get_iris_dataset


def test_logic_port(rn, test_data, title):
    print(title)
    for data in test_data:
        value = str(rn.process(data)[0])
        print("(" + str(data[0]) + "," + str(data[1]) + "): " + value)

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
    test_labels = [0, 0, 0, 1]
    test_labels_or = [0, 1, 1, 1]

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
    func_perceptron = lambda x: 1 if x > 0 else 0

    train_data = []
    train_labels = []
    for i in range(0, 100):
        index = random.randint(0, 3)
        train_data.append(test_data[index])
        train_labels.append(test_labels[index])

    rn_test = Perceptron(func_perceptron, 2, 0.1, 0.15)
    rn_test.train(train_data, train_labels)

    test_logic_port(rn_test, test_data, "==== Porta Lógica AND Perceptron ====")

    for i in range(0, 100):
        index = random.randint(0, 3)
        train_data.append(test_data[index])
        train_labels.append(test_labels_or[index])

    rn_test_or = Perceptron(func_perceptron, 2, 0.1, 0.10)
    rn_test_or.train(train_data, train_labels)
    test_logic_port(rn_test_or, test_data, "==== Porta Lógica OR Perceptron ====")

    # Run on Iris Dataset
    i# ris_dataset, iris_classes = get_iris_dataset("inputs/iris_train.json")
    # iris_test, iris_test_classes = get_iris_dataset("inputs/iris_test.json")
    # iris_dataset = np.matrix(iris_dataset).transpose().tolist()
    # iris_test = np.matrix(iris_test).transpose().tolist()

    # rn_iris = Perceptron(func_perceptron, 4, 0.1, 0.1)
    # rn_iris.train(iris_dataset, iris_classes)
    # test_logic_port(rn_iris, iris_test, "Teste iris")
