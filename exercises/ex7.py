from library.MLP import *
from library.RNA_Validation import *
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

    # Gets the best model
    cross_validation(iris_data, iris_classes, [iris_1, iris_2, iris_3, iris_4, iris_5], 5000)

