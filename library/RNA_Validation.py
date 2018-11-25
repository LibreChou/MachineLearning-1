import numpy as np


def shuffle_data(data, classes):
    permutation = np.random.permutation(len(data))

    data = [data[i] for i in permutation]
    classes = [classes[i] for i in permutation]

    return data, classes

def find_max_value_index(values):
    max_value = max(values)
    return values.index(max_value)

def get_accuracy(rna, dataset, classes):
    # Gets it accuracy in validation dataset
    acc = 0
    for i in range(0, len(dataset)):
        result = rna.process(dataset[i])

        # Compares if result is correct
        if find_max_value_index(result) == find_max_value_index(classes[i]):
            acc += 1

    return float(acc) / float(len(dataset))

def train_rna(rna, dataset, classes, epochs):
    for i in range(0, epochs):
        dataset, classes = shuffle_data(dataset, classes)
        rna.train(dataset, classes)

def cross_validation(data, classes, rnas, epochs):
    last_index_train = int(len(data) / 2)
    last_index_validation = last_index_train + int(len(data) / 4)
    last_index_test = len(data)

    train_dataset = [data[i] for i in range(0, last_index_train)]
    train_classes = [classes[i] for i in range(0, last_index_train)]

    validation_dataset = [data[i] for i in range(last_index_train, last_index_validation)]
    validation_classes = [classes[i] for i in range(last_index_train, last_index_validation)]

    test_dataset = [data[i] for i in range(last_index_validation, last_index_test)]
    test_classes = [classes[i] for i in range(last_index_validation, last_index_test)]


    accuracies = []
    # Validates RNA models
    for rna in rnas:
        # Trains for validation dataset
        train_rna(rna, train_dataset, train_classes, epochs)

        # Gets it accuracy in validation dataset
        acc = get_accuracy(rna, validation_dataset, validation_classes)
        accuracies.append(acc)

        print("RNA \"" + rna.name + "\" got an accuracy value in validation step of: " + str(acc))

    # Selects the best RNA
    best_rna = rnas[find_max_value_index(accuracies)]

    # Tests the best RNA in test dataset
    acc = get_accuracy(best_rna, test_dataset, test_classes)
    print("The best RNA is \"" + rna.name + "\" with an accuracy value in test step of:" + str(acc))

    return best_rna
