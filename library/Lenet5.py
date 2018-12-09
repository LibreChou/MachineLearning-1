import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


class Lenet5(object):

    def __init__(self):

        # Defines Lenet5 model
        self.model = Sequential([
            # The first layer is a convolution layer with 6 5x5 kernel
            # The input must be specified for the first layer which is a 28x28 image
            Conv2D(6, (5, 5), input_shape=(28, 28, 1), activation='relu'),
            # 2x2 Pooling layer
            MaxPooling2D((2, 2)),
            # 16 5x5 kernels
            Conv2D(16, (5, 5), activation='relu'),
            MaxPooling2D(2),
            # The flatten layer is used to convert input in an one dimensional array for fully connected layers
            Flatten(),
            # 30% of the input will be dropped
            Dropout(0.30),
            # Fully connected network with 120 Neurons with ReLU as activation function
            Dense(120, activation='relu'),
            # 40% of the input will be dropped
            Dropout(0.40),
            # Fully connected network with 84 Neurons with ReLU as activation function
            Dense(84, activation='relu'),
            # Gets the argmax and returns the classified label vector
            Dense(10, activation='softmax')
        ])

        # Defines learning parameters
        self.model.compile(optimizer=keras.optimizers.Adadelta(), loss=keras.losses.categorical_crossentropy,
                           metrics=['accuracy'])

        # Properties for after training
        self.loss = 0.0
        self.accuracy = 0.0

        # Loads database
        (self.train_data, self.train_labels), (self.test_data, self.test_labels) = mnist.load_data()

        # Ensures data is in the right shape
        self.train_data = self.train_data.reshape(self.train_data.shape[0], 28, 28, 1)
        self.test_data = self.test_data.reshape(self.test_data.shape[0], 28, 28, 1)

        # Normalizes data (maps it between 0 and 1)
        self.train_data = self.train_data.astype('float32')
        self.test_data = self.test_data.astype('float32')
        self.train_data /= 255
        self.test_data /= 255

        # Creates categorical binary classes (eg: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0] means the digit 1)
        self.train_labels = keras.utils.to_categorical(self.train_labels, 10)
        self.test_labels = keras.utils.to_categorical(self.test_labels, 10)

    def train(self, epochs, batch_size=128):
        # Trains model
        self.model.fit(self.train_data, self.train_labels, batch_size=batch_size, epochs=epochs, verbose=1,
                       validation_data=(self.test_data, self.test_labels))
        # evaluate it on test dataset
        results = self.model.evaluate(self.test_data, self.test_labels, verbose=0)
        # Update those properties according to last classification
        self.loss = results[0]
        self.accuracy = results[1]
