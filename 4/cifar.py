#!/bin/python3.6

import numpy as np
import tensorflow as tf
from time import time

# HYPERPARAMS
KERNEL_SIZE = 3
FILTERS = 32
DROPOUT = 0.4
L2 = 0.001
HEIGHT = 32
WIDTH = 32
DEPTH = 3
LEARN_RATE = 0.1

class CIFARData(object):
    def __init__(self, mode=10):
        """
        Images and Labels initialization
        """
        np.random.seed(31415)
        if mode == 10:
            data = tf.keras.datasets.cifar10
            self.num_classes = 10
        else:
            data = tf.keras.datasets.cifar100
            self.num_classes = 100
        (self.trainingImages, self.trainingLabels), (self.testImages, self.testLabels) = data.load_data()
        self.trainingImages, self.testImages = self.trainingImages / 255.0, self.testImages / 255.0
        self.train_feature, self.train_label, self.valid_feature, self.valid_label = self.splitData(self.trainingImages, self.trainingLabels)
        self.testImages = np.reshape(self.testImages, [-1,HEIGHT,WIDTH,DEPTH])
        self.testLabels = tf.keras.utils.to_categorical(self.testLabels, self.num_classes)

    def splitData(self, feature, label):
        '''
        Separation of Training Data into Training and Validation Sets
        Training Samples and Validation Samples to closely model
        actual test set
        '''
        index = np.arange(feature.shape[0])
        np.random.shuffle(index)
        feature_rand = np.reshape(feature[index], [-1,HEIGHT,WIDTH,DEPTH])
        label_rand = tf.keras.utils.to_categorical(label[index], self.num_classes)
        val = int(feature.shape[0] * 0.2)
        return (feature_rand[val:], label_rand[val:],
            feature_rand[:val], label_rand[:val])


class Model(tf.Module):
    def __init__(self, mode=10):
        self.nn = tf.keras.Sequential()
        # A 10 indicates the model is for CIFAR10 while anything else is 100
        if mode == 10:
            self.num_classes = 10
            self.CIFAR()
            self.metrics = ['accuracy']
        else:
            self.num_classes = 100
            self.CIFAR()
            self.metrics = ['accuracy', 'top_k_categorical_accuracy']
        # Take a look at the model summary
        self.nn.summary()
        self.nn.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=self.metrics)

    def SmallNetwork(self, nn, filters, kernel, dr, l2=0.001):
        nn.add(tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel,
            padding='same', activation='elu',
            kernel_regularizer=tf.keras.regularizers.l2(l2)))
        nn.add(tf.keras.layers.BatchNormalization())
        nn.add(tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel,
            padding='same', activation='elu',
            kernel_regularizer=tf.keras.regularizers.l2(l2)))
        nn.add(tf.keras.layers.MaxPool2D(pool_size=2))
        nn.add(tf.keras.layers.Dropout(dr))

    def CIFAR(self):
        self.nn.add(tf.keras.layers.BatchNormalization(input_shape=(HEIGHT, WIDTH, DEPTH)))
        for i in range(3):
            self.SmallNetwork(self.nn, FILTERS * (2**i), KERNEL_SIZE, DROPOUT, L2)
        self.nn.add(tf.keras.layers.Flatten())
        self.nn.add(tf.keras.layers.Dense(self.num_classes,activation="softmax",
            kernel_initializer=tf.keras.initializers.VarianceScaling()))

    def evaluate(self, test_feature, test_label):
        self.nn.evaluate(test_feature, test_label, verbose=1)

    def __call__(self, feature, label, valid_feature, valid_label):
        '''
        Function to initiate training
        '''
        checkpointer = tf.keras.callbacks.TensorBoard(log_dir="logs\\{}".format(time()))
        self.nn.fit(feature,
            label,
            batch_size=512,
            epochs=100,
            validation_data=(valid_feature, valid_label),
            callbacks=[checkpointer])

if __name__ == "__main__":
    data = CIFARData(10)
    model = Model(10)
    model(data.train_feature, data.train_label, data.valid_feature, data.valid_label)
    # The Following Line was added only after modifying hyperparameters to check
    model.evaluate(data.testImages, data.testLabels)
