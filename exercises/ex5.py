from library.BayesNaive import BayesNaive
from sklearn.naive_bayes import GaussianNB

def ex1_dataset():
    values = [["PC"], ["PC"], ["PC"], ["Mac"], ["Mac"], ["Mac"], ["Mac"], ["Mac"], ["PC"], ["Mac"]]
    classes = ["Android", "Android", "Android", "iPhone", "Android", "iPhone", "iPhone", "Android", "iPhone", "iPhone"]
    return values, classes


def ex5():
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
    classes = ["No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No",
               "Yes", "Yes", "Yes", "Yes", "Yes", "No"]
    test = BayesNaive(tenis_dataset, classes)
    # print(test.get_discrete_probability(["Sunny", "Cool", "High", "Strong"]))

    # real exercise
    laptops, phones = ex1_dataset()
    bayes1 = BayesNaive(laptops, phones)
    prob_phone = bayes1.get_discrete_probability(["Mac"])
    print(prob_phone)

    # Teste Continuous
    temp_data = [[25.2], [19.3], [18.5], [21.7], [20.1], [24.3], [22.8], [23.1], [19.8], [27.3], [30.1], [17.4], [29.5],[15.1]]
    temp_classes = ["Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "No", "No", "No", "No", "No"]
    test_c = BayesNaive(temp_data, temp_classes, continuous=True)
    print(test_c.get_probability([25.5]))
    test = 18.2
    print(test_c.predict([test]))

    test_temp = GaussianNB()
    test_temp.fit(temp_data, temp_classes)
    print(test_temp.predict([[test]]))
    print(test_temp.predict_proba([[test]]))
