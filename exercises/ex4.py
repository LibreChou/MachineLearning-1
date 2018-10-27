import matplotlib.pyplot as plt
from library.kmeans import *
from library.LDA import *
import json


def plot_points(array, groups, black = False):
    if len(array) != 2:
        raise ValueError("Cannot plot not two-dimensional array")


    # get_groups
    unique_groups = []
    for i in range(0, len(groups)):
        if not groups[i] in unique_groups:
            unique_groups.append(groups[i])

    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    n = len(colors)
    c = 0


    # plot all groups
    for group in unique_groups:
        data = [[], []]
        for i in range(0, len(array[0])):
            if groups[i] == group:
                data[0].append(array[0][i])
                data[1].append(array[1][i])
        if black:
            plt.scatter(data[0], data[1], c='k')
        else:
            plt.scatter(data[0], data[1], c=colors[c % n])
            c += 1

def get_iris_dataset():
    # test data
    # array = [[1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 3.0, 5.0, 6.0],
    #         [2.0, 3.0, 3.0, 5.0, 5.0, 0.0, 1.0, 1.0, 2.0, 3.0, 5.0]]
    # classes = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]

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

def load_data(filepath):
    f = open(filepath, "r")
    content = f.read()
    content = content.split("\n")
    content = [cont.split(",") for cont in content]
    classes = [cont[0] for cont in content]
    data = [[float(line[i]) for i in range(1, len(line))] for line in content]
    data = np.matrix(data).transpose().tolist()
    return data, classes

def lda_and_kmeans(inputs, classes, title, xlabel, ylabel, exec_lda = False):
    # Gets LDA data
    if exec_lda:
        lda = LDA(inputs, classes)
        print(lda.eigen_values)
        print(lda.eigen_vectors)
        fv = lda.get_feature_vector(2)
        print(lda.eigen_values)
        print(lda.eigen_vectors)
        data = np.linalg.pinv(fv).dot(lda.samples).tolist()
        print(fv)
        plot_points(data, classes)
        plt.title("Only LDA")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid()
        plt.show()
    else:
        data = inputs

    # Executes iris
    kmeans = Kmeans(data, 3)
    kmeans.execute()
    groups = kmeans.get_groups(kmeans.data)
    centers = np.matrix(kmeans.center).transpose().tolist()
    plot_points(data, groups)
    plot_points(centers, len(centers[0]) * [1.0], True)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()

def ex4():
    # Exercise 1
    input_ex1 = [[1.9, 3.4, 2.5, 1.5, 3.5, 2.2, 3.4, 3.6, 5.0, 4.5, 6.0, 1.9, 1.0, 1.9, 0.8, 1.6, 1.0],
                 [7.3, 7.5, 6.8, 6.5, 6.4, 5.8, 5.2, 4.0, 3.2, 2.4, 2.6, 3.0, 2.7, 2.4, 2.0, 1.8, 1.0]]
    lda_and_kmeans(input_ex1, [], "K-Means", "X", "Y")

    # Run on Iris Dataset
    iris_dataset, classes = get_iris_dataset()
    lda_and_kmeans(iris_dataset, classes, "LDA + K-Means in IrisDatased", "DC1", "DC2", True)

    # Run on Iris Dataset
    wine_dataset, wine_classes = load_data("inputs/wine.txt")
    lda_and_kmeans(wine_dataset, wine_classes, "LDA + K-Means in WineDataset", "DC1", "DC2", True)

    # Run on Abalone Dataset
    abalone_dataset, abalone_classes = load_data("inputs/abalone.txt")
    lda_and_kmeans(abalone_dataset, abalone_classes, "LDA + K-Means in AbaloneDataset", "DC1", "DC2", True)



