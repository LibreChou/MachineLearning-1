from library.BayesNaive import BayesNaive
import numpy as np
import json
from sklearn.naive_bayes import GaussianNB

def get_iris_dataset(path):
    # Fills inputs
    array = []
    classes = []

    # Reads and parse Json
    with open(path) as f:
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


# Prints number of success and total
def get_performance(data, classes, test_values, test_classes, continuous=False):
    naive_bayes = BayesNaive(data, classes, continuous=continuous)
    total = 0

    for i in range(0, len(test_values)):
        classification = test_classes[i]
        prediction = naive_bayes.predict(test_values[i])[0]
        if classification == prediction:
            total += 1

    print("Acertou " + str(total) + " de " + str(len(test_values)))


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
    iris_dataset, iris_classes = get_iris_dataset("inputs/iris_train.json")
    iris_test, iris_test_classes = get_iris_dataset("inputs/iris_test.json")
    iris_dataset = np.matrix(iris_dataset).transpose().tolist()
    iris_test = np.matrix(iris_test).transpose().tolist()

    print("==== Iris DataSet ====")
    get_performance(iris_dataset, iris_classes, iris_test, iris_test_classes, True)

    # Run on wine Dataset
    wine_dataset, wine_classes = load_data("inputs/wine_train.txt")
    wine_test, wine_test_classes = load_data("inputs/wine_test.txt")
    wine_dataset = np.matrix(wine_dataset).transpose().tolist()
    wine_test = np.matrix(wine_test).transpose().tolist()
    print("==== Wine DataSet ====")
    get_performance(wine_dataset, wine_classes, wine_test, wine_test_classes, True)

    # Run on Abalone Dataset
    abalone_dataset, abalone_classes = load_data("inputs/abalone_train.txt")
    abalone_test, abalone_test_classes = load_data("inputs/abalone_test.txt")
    abalone_dataset = np.matrix(abalone_dataset).transpose().tolist()
    abalone_test = np.matrix(abalone_test).transpose().tolist()
    print("==== Abalone DataSet ====")
    get_performance(abalone_dataset, abalone_classes, abalone_test, abalone_test_classes, True)
