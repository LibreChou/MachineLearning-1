from library.BayesNaive import BayesNaive
import numpy as np
import json
from sklearn.naive_bayes import GaussianNB
from exercises.ex4 import load_data

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
        classes.append(c)

    return array, classes

def ex1_dataset():
    values = [["PC"], ["PC"], ["PC"], ["Mac"], ["Mac"], ["Mac"], ["Mac"], ["Mac"], ["PC"], ["Mac"]]
    classes = ["Android", "Android", "Android", "iPhone", "Android", "iPhone", "iPhone", "Android", "iPhone", "iPhone"]
    return values, classes


def exec_experiment(data, classes, test_value, title, continuous=False):
    print(title)

    # Exec Implemented Naive Bayes
    naive_bayes = BayesNaive(data, classes, continuous=continuous)
    result = naive_bayes.predict(test_value)
    print("Implemented code Prediction - Class: " + result[0].__str__()
          + "; Probability: " + result[1].__str__())

    # Tests against SkLearn Implementation
    if continuous:
        naive_bayes = GaussianNB()
        value = []
        value.append(test_value)
        naive_bayes.fit(data, classes)
        print("Sklearn code Prediction - Class: " + naive_bayes.predict(value).__str__()
              + "; Probability: " + naive_bayes.predict_proba(value).__str__())

    print()

def ex5():
    # Datasets
    tenis_dataset = [
        ["Sunny", "Hot", "High", "Weak"],  # D1
        ["Sunny", "Hot", "High", "Strong"],  # D2
        ["Overcast", "Hot", "High", "Weak"],  # D3
        ["Rain", "Mild", "High", "Weak"],  # D4
        ["Rain", "Cool", "Normal", "Weak"],  # D5
        ["Rain", "Cool", "Normal", "Strong"],  # D6
        ["Overcast", "Cool", "Normal", "Strong"],  # D7
        ["Sunny", "Mild", "High", "Weak"],  # D8
        ["Sunny", "Cool", "Normal", "Weak"],  # D9
        ["Rain", "Mild", "Normal", "Weak"],  # D10
        ["Sunny", "Mild", "Normal", "Strong"],  # D11
        ["Overcast", "Mild", "High", "Strong"],  # D12
        ["Overcast", "Hot", "Normal", "Weak"],  # D13
        ["Rain", "Mild", "High", "Strong"]  # D14
    ]
    tenis_classes = ["No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No",
               "Yes", "Yes", "Yes", "Yes", "Yes", "No"]

    temp_data = [[25.2], [19.3], [18.5], [21.7], [20.1], [24.3], [22.8], [23.1], [19.8], [27.3], [30.1], [17.4], [29.5],[15.1]]
    temp_classes = ["Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "No", "No", "No", "No", "No"]

    # Tests
    exec_experiment(tenis_dataset, tenis_classes, ["Sunny", "Cool", "High", "Strong"], "==== Tenis Exercise ====")
    exec_experiment(temp_data, temp_classes, [25.2], "==== Temperature Exercise ====", True)

    # Exercises
    dataset_ex1, classes_ex1 = ex1_dataset()
    exec_experiment(dataset_ex1, classes_ex1, ["Mac"], "==== Ex1 ====")

    # Run on Iris Dataset
    iris_dataset, iris_classes = get_iris_dataset()
    iris_dataset = np.matrix(iris_dataset).transpose().tolist()
    exec_experiment(iris_dataset, iris_classes, [7.0, 3.3, 5.4, 0.3], "==== Iris Dataset ====", True)

    # Run on winde Dataset
    wine_dataset, wine_classes = load_data("inputs/wine.txt")
    wine_dataset = np.matrix(wine_dataset).transpose().tolist()
    exec_experiment(wine_dataset, wine_classes, [13.28, 1.64, 2.84, 25.5, 120, 2.8, 3.01, .34, 1.56, 4.0, 1.09, 2.78,
                                                 910.0], "==== Wine Dataset 1 ====", True)
    exec_experiment(wine_dataset, wine_classes, [12.99, 1.67, 2.6, 36.0, 119.0, 3.3, 2.70, .21, 1.56, 3.1, 1.31, 3.5,
                                                 960.0], "==== Wine Dataset 2 ====", True)

    # Run on Abalone Dataset
    abalone_dataset, abalone_classes = load_data("inputs/abalone.txt")
    abalone_dataset = np.matrix(abalone_dataset).transpose().tolist()
    exec_experiment(abalone_dataset, abalone_classes, [0.425, 0.36, 0.087, 0.570, 0.2545, 0.112, 0.15, 9.0], "==== Abalone Dataset 1 - M ====", True)
    exec_experiment(abalone_dataset, abalone_classes, [0.6, 0.42, 0.110, 0.635, 0.2065, 0.1115, 0.21, 9.3], "==== Abalone Dataset 2 - F ====", True)

