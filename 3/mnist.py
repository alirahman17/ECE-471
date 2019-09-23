#!/bin/python3.6

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from time import time

from tqdm import trange

# HYPERPARAMS
KERNEL_SIZE = 5
POOL_SIZE = 4
FILTER = 3

class MNISTData(object):
    def __init__(self):
        """
        The two spirals data initialization
        """
        self.trainingImages, self.trainingLabels = self.loadMNIST("train", "./")
        self.testImages, self.testLabels = self.loadMNIST("t10k", "./")
        self.trainingImages, self.testImages = self.trainingImages / 255.0, self.testImages / 255.0
        self.train_feature, self.train_label, self.valid_feature, self.valid_label = self.splitData(self.trainingImages, self.trainingLabels)
        self.testImages = np.reshape(self.testImages, [-1,28,28,1])
        self.testLabels = tf.keras.utils.to_categorical(self.testLabels, 10)


    def loadMNIST(self, prefix, folder):
        intType = np.dtype( 'int32' ).newbyteorder( '>' )
        nMetaDataBytes = 4 * intType.itemsize
        data = np.fromfile( folder + "/" + prefix + '-images-idx3-ubyte', dtype = 'ubyte' )
        magicBytes, nImages, width, height = np.frombuffer( data[:nMetaDataBytes].tobytes(), intType )
        data = data[nMetaDataBytes:].astype( dtype = 'float32' ).reshape( [ nImages, width, height ] )
        labels = np.fromfile( folder + "/" + prefix + '-labels-idx1-ubyte',
                              dtype = 'ubyte' )[2 * intType.itemsize:]
        return data, labels

    def splitData(self, feature, label):
        index = np.arange(feature.shape[0])
        np.random.shuffle(index)
        feature_rand = np.reshape(feature[index], [-1,28,28,1])
        label_rand = tf.keras.utils.to_categorical(label[index], 10)
        val = int(feature.shape[0] * 0.2)
        return (feature_rand[val:], label_rand[val:],
            feature_rand[:val], label_rand[:val])


class Model(tf.Module):
    def __init__(self):
        height, width = 28, 28
        self.nn = tf.keras.Sequential()
        # Must define the input shape in the first layer of the neural network
        self.nn.add(tf.keras.layers.Conv2D(filters=FILTER, kernel_size=KERNEL_SIZE, padding='same', activation='elu', input_shape=(28,28,1)))
        self.nn.add(tf.keras.layers.MaxPooling2D(pool_size=POOL_SIZE))
        self.nn.add(tf.keras.layers.Dropout(0.08))
        self.nn.add(tf.keras.layers.Flatten())
        self.nn.add(tf.keras.layers.Dense(10, activation='softmax'))
        # Take a look at the model summary
        self.nn.summary()
        self.nn.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

    def evaluate(self, test_feature, test_label):
        self.nn.evaluate(test_feature, test_label)

    def __call__(self, feature, label, valid_feature, valid_label):
        checkpointer = tf.keras.callbacks.TensorBoard(log_dir="logs\\{}".format(time()))
        self.nn.fit(feature,
         label,
         batch_size=1024,
         epochs=50,
         validation_data=(valid_feature, valid_label),
         callbacks=[checkpointer])

if __name__ == "__main__":
    data = MNISTData()
    model = Model()
    model(data.train_feature, data.train_label, data.valid_feature, data.valid_label)
    # The Following Line was added after modifying hyperparameters
    model.evaluate(data.testImages, data.testLabels)
