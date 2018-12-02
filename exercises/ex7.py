from library.MLP import *
from library.RNA_Validation import *
from sklearn import preprocessing
import json

def get_iris_dataset(path):
    # Fills inputs
    array = []
    classes = []

    # Reads and parse Json
    with open(path) as f:
        data = json.load(f)

    properties = ["sepalLength", "sepalWidth", "petalLength", "petalWidth"]
    species = ["setosa", "versicolor", "virginica"]

    for d in data:
        array.append([float(d[p]) for p in properties])
        d_classes = [1 if d["species"] == specie else 0 for specie in species]
        classes.append(d_classes)

    return array, classes


def load_data(filepath, labels):
    f = open(filepath, "r")
    content = f.read()
    content = content.split("\n")
    content = [cont.split(",") for cont in content]
    classes = [cont[0] for cont in content]
    classes = [[1 if c == label else 0 for label in labels] for c in classes]
    data = [[float(line[i]) for i in range(1, len(line))] for line in content]
    return data, classes


def ex7():

    # Test many Topologies on Iris Data Set
    iris_1 = MLP("Iris: 3", 0.1, 0.1, 4, [3])
    iris_2 = MLP("Iris: 3-3", 0.1, 0.1, 4, [3, 3])
    iris_3 = MLP("Iris: 4-3", 0.1, 0.1, 4, [4, 3])
    iris_4 = MLP("Iris: 4-3-3", 0.1, 0.1, 4, [4, 3, 3])
    iris_5 = MLP("Iris: 4-3-3-3", 0.1, 0.1, 4, [4, 3, 3, 3])

    # Gets data set
    iris_data, iris_classes = get_iris_dataset("inputs/iris.json")
    iris_data, iris_classes = shuffle_data(iris_data, iris_classes)
    iris_data = preprocessing.normalize(iris_data)

    # Gets the best model
    print("==== Cross Validation for Iris Data Set ====")
    cross_validation(iris_data, iris_classes, [iris_1, iris_2, iris_3, iris_4, iris_5], 12000)
    print()

    # Test many Topologies on Parkinson Data Set
    pk_1 = MLP("Parkinson: 2", 0.1, 0.1, 22, [2])
    pk_2 = MLP("Parkinson: 4-2", 0.1, 0.1, 22, [4, 2])
    pk_3 = MLP("Parkinson: 3-2", 0.1, 0.1, 22, [3, 2])
    pk_4 = MLP("Parkinson: 3-5-2", 0.1, 0.1, 22, [3, 5, 2])

    pk_data, pk_classes = load_data("inputs/parkinson.txt", ['0', '1'])
    pk_data, pk_classes = shuffle_data(pk_data, pk_classes)
    pk_data = preprocessing.normalize(pk_data)

    # Gets the best model
    print("==== Cross Validation for Parkinson Data Set ====")
    cross_validation(pk_data, pk_classes, [pk_1, pk_2, pk_3, pk_4], 12000)
    print()

    # Test many Topologies on wine Data Set
    wine_1 = MLP("Wine: 3", 0.1, 0.1, 13, [3])
    wine_2 = MLP("Wine: 4-3", 0.1, 0.1, 13, [4, 3])
    wine_3 = MLP("Wine: 4-3-3", 0.1, 0.1, 13, [4, 3, 3])
    wine_4 = MLP("Wine: 4-5-5-3", 0.1, 0.1, 13, [4, 5, 5, 3])

    # Gets data set
    wine_data, wine_classes = load_data("inputs/wine.txt", ['1', '2', '3'])
    wine_data, wine_classes = shuffle_data(wine_data, wine_classes)
    wine_data = preprocessing.normalize(wine_data, copy=True)

    # Gets the best model
    print("==== Cross Validation for Wine Data Set ====")
    cross_validation(wine_data, wine_classes, [wine_1, wine_2, wine_3, wine_4], 12000)
    print()
