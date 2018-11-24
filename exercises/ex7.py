from library.MLP import *
from library.Neuron import *
import json
import random

def get_iris_dataset(path):
    # Fills inputs
    array = []
    classes = []

    # Reads and parse Json
    with open(path) as f:
        data = json.load(f)
    random.shuffle(data)
    properties = ["sepalLength", "sepalWidth", "petalLength", "petalWidth"]
    species = ["setosa", "versicolor", "virginica"]

    for d in data:
        array.append([float(d[p]) for p in properties])
        d_classes = [1 if d["species"] == specie else 0 for specie in species]
        classes.append(d_classes)

    return array, classes


def ex7():
    f_ativ = lambda x: 1.0 / (1 + np.power(np.e, -x))


    test = MLP(0.1, 0.1, 2)
    test.set_layer(0, [Neuron([.1, .1], f_ativ), Neuron([.2, .2], f_ativ)])
    test.set_layer(1, [Neuron([.1, .2], f_ativ)])
    test.make_matrix()
    test.train([[1.0, 2.0]], [[1.0]])


    rna = MLP(0.1, 0.1, 3)
    rna.set_layer(0, [Neuron([.1, .2, .3, .4], f_ativ), Neuron([.1, .2, .3, .4], f_ativ), Neuron([.1, .2, .3, .4], f_ativ)])
    rna.set_layer(1, [Neuron([.1, .2, .3], f_ativ), Neuron([.1, .2, .3], f_ativ), Neuron([.1, .2, .3], f_ativ)])
    rna.set_layer(2, [Neuron([.1, .2, .3], f_ativ), Neuron([.1, .2, .3], f_ativ),
                      Neuron([.1, .2, .3], f_ativ)])
    rna.init_random_weights()
    rna.make_matrix()

    test_data = [7.9, 3.8, 6.4, 2.0]
    print(rna.process(test_data))
    iris_data, iris_classes = get_iris_dataset("inputs/iris_train.json")
    iris_test, iris_test_classes = get_iris_dataset("inputs/iris_test.json")
    for i in range(0, 100):
        # data_to_shuffle = list(zip(iris_data, iris_classes))
        # random.shuffle(data_to_shuffle)
        # iris_data, iris_classes = zip(*data_to_shuffle)
        rna.train(iris_data, iris_classes, True)

    for i in range(0, len(iris_test)):
        result = rna.process(iris_test[i])
        print("Resultado: " + str(result) + "\nResultado Esperado: " + str(iris_test_classes[i]))
