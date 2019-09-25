#!/bin/python3.6

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from time import time

from tqdm import trange

# HYPERPARAMS (These were eventually ignored)
KERNEL_SIZE = 5
POOL_SIZE = 4
FILTER = 3

class MNISTData(object):
    def __init__(self):
        """
        Images and Labels initialization
        """
        self.trainingImages, self.trainingLabels = self.loadMNIST("train")
        self.testImages, self.testLabels = self.loadMNIST("t10k")
        self.trainingImages, self.testImages = self.trainingImages / 255.0, self.testImages / 255.0
        self.train_feature, self.train_label, self.valid_feature, self.valid_label = self.splitData(self.trainingImages, self.trainingLabels)
        self.testImages = np.reshape(self.testImages, [-1,28,28,1])
        self.testLabels = tf.keras.utils.to_categorical(self.testLabels, 10)

    def loadMNIST(self, prefix):
        '''
        Function to load all of MNIST data into separate images and labels
        '''
        intType = np.dtype('int32').newbyteorder('>')
        nMeta = 4 * intType.itemsize
        data = np.fromfile("./" + prefix + '-images-idx3-ubyte', dtype = 'ubyte')
        _, n, w, h = np.frombuffer(data[:nMeta].tobytes(), intType)
        data = data[nMeta:].astype(dtype = 'float32').reshape([n, w, h])
        labels = np.fromfile("./" + prefix + '-labels-idx1-ubyte',
                              dtype = 'ubyte')[2 * intType.itemsize:]
        return data, labels

    def splitData(self, feature, label):
        '''
        Separation of Training Data into Training and Validation Sets
        50000 Training Samples and 10000 Validation Samples to closely model
        actual test set
        '''
        index = np.arange(feature.shape[0])
        np.random.shuffle(index)
        feature_rand = np.reshape(feature[index], [-1,28,28,1])
        label_rand = tf.keras.utils.to_categorical(label[index], 10)
        val = int(feature.shape[0] * 0.16667)
        return (feature_rand[val:], label_rand[val:],
            feature_rand[:val], label_rand[:val])


class Model(tf.Module):
    def __init__(self, mode):
        self.nn = tf.keras.Sequential()
        # Must define the input shape in the first layer of the neural network
        # A 1 indicates high accuracy While anything wlse indicated low parameters
        if mode == 1:
            self.highAccuracy()
        else:
            self.lowParameters()
        # Take a look at the model summary
        self.nn.summary()
        self.nn.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

    def highAccuracy(self):
        self.nn.add(tf.keras.layers.Conv2D(filters=4, kernel_size=6, strides=1,
            padding='valid', activation='elu', input_shape=(28,28,1),
            kernel_regularizer=tf.keras.regularizers.l2(l=0.001)))
        self.nn.add(tf.keras.layers.MaxPool2D(pool_size=POOL_SIZE))
        self.nn.add(tf.keras.layers.Dropout(0.05))
        '''
        Changing the Filters in the following line to 3 guarantees 96% accuracy
        but with 689 parameters.
        I attempted to minimize the parameters with at least 95.5% accuracy
        so I have left it at 2 to achieve 512 parameters with 95.5% accuracy
        most of the time.
        '''
        self.nn.add(tf.keras.layers.Conv2D(filters=2, kernel_size=2, strides=1,
            padding='valid', activation='elu',
            kernel_regularizer=tf.keras.regularizers.l2(l=0.001)))
        self.nn.add(tf.keras.layers.Flatten())
        self.nn.add(tf.keras.layers.Dense(10,activation="softmax",
            kernel_regularizer=tf.keras.regularizers.l2(l=0.001)))

    def lowParameters(self):
        self.nn.add(tf.keras.layers.Conv2D(filters=2, kernel_size=6, strides=1,
            padding='valid', activation='elu', input_shape=(28,28,1)))
        self.nn.add(tf.keras.layers.MaxPool2D(pool_size=POOL_SIZE))
        self.nn.add(tf.keras.layers.Dropout(0.05))
        self.nn.add(tf.keras.layers.Conv2D(filters=1, kernel_size=2, strides=1,
            padding='valid', activation='relu'))
        self.nn.add(tf.keras.layers.Dropout(0.05))
        self.nn.add(tf.keras.layers.Conv2D(filters=1, kernel_size=2, strides=2,
            padding='valid', activation='elu'))
        self.nn.add(tf.keras.layers.Flatten())
        self.nn.add(tf.keras.layers.Dense(10,activation="softmax",
            kernel_regularizer=tf.keras.regularizers.l2(l=0.01)))

    def evaluate(self, test_feature, test_label):
        self.nn.evaluate(test_feature, test_label)

    def __call__(self, feature, label, valid_feature, valid_label):
        '''
        Function to initiate training
        '''
        checkpointer = tf.keras.callbacks.TensorBoard(log_dir="logs\\{}".format(time()))
        self.nn.fit(feature,
            label,
            batch_size=1024,
            epochs=200,
            validation_data=(valid_feature, valid_label),
            callbacks=[checkpointer])

if __name__ == "__main__":
    data = MNISTData()
    model = Model(1)
    model(data.train_feature, data.train_label, data.valid_feature, data.valid_label)
    # The Following Line was added only after modifying hyperparameters to check
    model.evaluate(data.testImages, data.testLabels)
