from library.Lenet5 import *
from library.Model_Evaluation import *


# Executes 1st proposed experiment
def exec_lenet_5():
    # Initializes Lenet5 model
    lenet_5 = Lenet5()
    # Trains it for 12 epochs and evaluates it
    lenet_5.train(12)
    # Gets it accuracy
    print("Lenet5 experiment had an accuracy of %.4f in mnist dataset!" % lenet_5.accuracy)


# Executes 2nd proposed experiment
def image_net_experiment():
    # Tests Xception CNN on dataset
    xception = keras.applications.xception.Xception(weights='imagenet')
    Model_Evaluation.get_net_accuracy("Xception", xception, (224, 224))

    # Tests InceptionResNetV2 CNN on dataset
    inception = keras.applications.inception_resnet_v2.InceptionResNetV2(weights='imagenet')
    Model_Evaluation.get_net_accuracy("InceptionResNetV2", inception, (299, 299))

    # Tests NasNetLarge CNN on dataset
    nasnet = keras.applications.nasnet.NASNetLarge(weights='imagenet')
    Model_Evaluation.get_net_accuracy("NasNetLarge", nasnet, (331, 331))


def ex8():

    exec_lenet_5()
    image_net_experiment()
