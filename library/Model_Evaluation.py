from library.Lenet5 import *
import numpy as np
import glob
from keras.preprocessing import image


# This class was implemented to help on executing exercise's 2 experiments
class Model_Evaluation(object):

    # Evaluates a CNN in the exercise's 2 database
    @staticmethod
    def get_net_accuracy(name, model, image_size):
        # Gets dataset with Huskies, Cats and Robins images and corresponding labels
        images, labels = Model_Evaluation.get_ex2_dataset(image_size)
        # Sets its training and evaluating configurations
        model.compile(optimizer=keras.optimizers.Adadelta(), loss=keras.losses.sparse_categorical_crossentropy,
                      metrics=['accuracy'])
        # Evaluates the model
        results = model.evaluate(images, labels, verbose=0)
        print("Model \"" + name + "\" - got and accuracy of %.4f" % results[1])

    # Returns all Huskies, Cats and Robins images and corresponding labels
    @staticmethod
    def get_ex2_dataset(image_size):
        # Concatenates all images in one ndarray
        images = np.concatenate((Model_Evaluation.get_image_dataset("inputs/Robin", image_size),
                                 Model_Evaluation.get_image_dataset("inputs/Cat", image_size),
                                 Model_Evaluation.get_image_dataset("inputs/Husky", image_size)), axis=0)

        # Creates ndarray with all labels (500 of each)
        # Those labels refer's to the indexes in the ImageNet's JSON labels representation
        # The JSON representation can be accessed by the url: https://github.com/raghakot/keras-vis/blob/master/resources/imagenet_class_index.json
        labels = np.concatenate(([15] * 500, [285] * 500, [248] * 500))
        return images, labels

    # Loads and pre_process images
    @staticmethod
    def get_image_dataset(path, image_size):
        # Gets all bmp images in the folder
        images = glob.glob(path + "/*.bmp")
        # Reads and resize image to desired shape
        images = [image.load_img(img, target_size=image_size) for img in images]
        # Converts each image in a numpy array object (ndarray)
        images = [image.img_to_array(img) for img in images]
        # Converts the whole image collection in a ndarray
        images = np.asarray(images)
        # Converts 8 bit integer pixel values in float32 values
        images = images.astype('float32')
        # Normalizes inputs to a range between 0 and 1
        images /= 255

        return images
