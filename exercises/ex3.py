import matplotlib.pyplot as plt
from library.LDA import *
from library.PCA import *
import json


def get_iris_dataset():
    # Fills inputs
    array = []
    classes = []

    # Reads and parse Json
    with open("inputs/iris.json") as f:
        data = json.load(f)

    # Apprend variables
    array.append([])
    for d in data:
        array[0].append(float(d["sepalLength"]))
    array.append([])
    for d in data:
        array[1].append(float(d["sepalWidth"]))
    array.append([])
    for d in data:
        array[2].append(float(d["petalLength"]))
    array.append([])
    for d in data:
        array[3].append(float(d["petalWidth"]))
    for d in data:
        c = d["species"]
        if c == "setosa":
            classes.append(1)
        elif c == "versicolor":
            classes.append(2)
        else:
            classes.append(3)

    return array, classes

def group_iris_type(array, classes, type):
    # Group Setosas
    group = [[], []]
    for iris in range(0, len(array[0])):
        if classes[iris] == type:
            group[0].append(array[0][iris])
            group[1].append(array[1][iris])
    return group

def plot_iris_exercise(array, classes, title, xlabel, ylabel):
    # Group Irises
    setosas = group_iris_type(array, classes, 1)
    versicolor = group_iris_type(array, classes, 2)
    virginica = group_iris_type(array, classes, 3)

    # Colors for classes. Blue for setora; Red for versicolor and Green for virgicina
    plt.scatter(setosas[0], setosas[1], c='b', label="Setosas")
    plt.scatter(versicolor[0], versicolor[1], c='r', label="Versicolor")
    plt.scatter(virginica[0], virginica[1], c='g', label="Virginica")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # hide axis ticks
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
                    labelbottom="on", left="off", right="off", labelleft="on")
    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)

    plt.grid()
    plt.show()

def ex3():

    array, classes = get_iris_dataset()
    #array = [[1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 3.0, 5.0, 6.0],
    #         [2.0, 3.0, 3.0, 5.0, 5.0, 0.0, 1.0, 1.0, 2.0, 3.0, 5.0]]
    #classes = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]


    l1 = LDA(array, classes)
    print(np.matrix(l1.scatter_between))
    print(np.matrix(l1.scatter_within))
    print(l1.eigen_values)
    print(l1.eigen_vectors)
    print(l1.get_feature_vector(2))

    data = np.linalg.pinv(l1.get_feature_vector(2)).dot(l1.samples).tolist()
    plot_iris_exercise(data, classes, "Iris Dataset com duas discriminantes após aplicação do LDA", "DC 1", "DC 2")

    # applies pca
    pca1 = PCA(array)
    pca1.set_precision(.93)

    fdata = pca1.get_final_data().tolist()
    plot_iris_exercise(fdata, classes, "PCA aplicado ao iris dataset", "PC 1", "PC 2")

    l2 = LDA(fdata, classes)
    print(np.matrix(l2.scatter_between))
    print(np.matrix(l2.scatter_within))
    print(l2.eigen_values)
    print(l2.eigen_vectors)

    data2 = np.linalg.pinv(l2.get_feature_vector(2)).dot(l2.samples).tolist()
    plot_iris_exercise(array, classes, "PCA + LDA", "DC 1", "DC 2")


    #print(l1.classes)