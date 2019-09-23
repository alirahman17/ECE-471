#!/bin/python3.6

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import trange

NUM_POINTS = 500
BATCH_SIZE = 100
NUM_BATCHES = 20000
LEARNING_RATE = 0.1
LAMBDA = 0.001

class SpiralData(object):
    def __init__(self, num_points=NUM_POINTS):
        """
        The two spirals data initialization
        """
        self.x, self.y, self.label = self.spiral_gen(num_points)

    def spiral_gen(self, num_points):
        sigma = 0.25
        noise_x1 = sigma*np.random.normal(size=(num_points, 1))
        noise_y1 = sigma*np.random.normal(size=(num_points, 1))
        noise_x2 = sigma*np.random.normal(size=(num_points, 1))
        noise_y2 = sigma*np.random.normal(size=(num_points, 1))
        np.random.seed(31415)

        # Separation of the spirals for ease of plotting later
        self.index = np.arange(num_points * 2)
        self.theta = np.sqrt(np.random.rand(num_points,1)) * 780 * (2*np.pi)/360
        self.x1 = -np.cos(self.theta)*self.theta + noise_x1
        self.x2 = np.cos(self.theta)*self.theta + noise_x2
        self.y1 = np.sin(self.theta)*self.theta + noise_y1
        self.y2 = -np.sin(self.theta)*self.theta + noise_y2

        return np.concatenate((self.x1,self.x2)), np.concatenate((self.y1,
            self.y2)), np.concatenate((np.zeros(num_points), np.ones(num_points)))

    def get_batch(self, batch_size=BATCH_SIZE):
        """
        Select random subset of examples for training batch
        """
        choices = np.random.choice(self.index, size=batch_size)
        return self.x[choices], self.y[choices], self.label[choices]


class Model(tf.Module):
    def __init__(self, l1, l2):
        # Weights between each of the layers
        # Input to Hidden Layer 1
        self.w1 = tf.Variable(tf.random.normal(shape=[2, l1]))
        # Hidden Layer 1 to Hidden Layer 2
        self.w2 = tf.Variable(tf.random.normal(shape=[l1, l2]))
        # Hidden Layer 2 to Output
        self.w3 = tf.Variable(tf.random.normal(shape=[l2, 1]))

        # Biases between each layer
        self.b1 = tf.Variable(tf.random.normal(shape=[1, l1]))
        self.b2 = tf.Variable(tf.random.normal(shape=[1, l2]))
        self.b3 = tf.Variable(tf.random.normal(shape=[1, 1]))

    def __call__(self, x, y):
        # This is the functional form of f(x)
        hidden_layer1 = tf.nn.elu((np.hstack((x, y)) @ self.w1) + self.b1)
        hidden_layer2 = tf.nn.elu((hidden_layer1 @ self.w2) + self.b2)
        output_layer = (hidden_layer2 @ self.w3) + self.b3
        # No need to call sigmoid as that is called on the cross entropy function
        return output_layer

if __name__ == "__main__":
    spiral = SpiralData()
    model = Model(45, 63)
    optimizer = tf.optimizers.SGD(learning_rate=LEARNING_RATE)
    bar = trange(NUM_BATCHES)
    for i in bar:
        with tf.GradientTape() as tape:
            x, y, label = spiral.get_batch()
            label_hat = model(x, y)
            label_s = np.float32(np.reshape(label, (BATCH_SIZE, 1)))
            # Loss function is a combination of sigmoid cross entropy and L2 Loss
            loss = (tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=label_s, logits=label_hat)) +
                tf.reduce_sum(0.001*tf.nn.l2_loss(model.w1) +
                LAMBDA*tf.nn.l2_loss(model.w2) +
                LAMBDA*tf.nn.l2_loss(model.w3) +
                LAMBDA*tf.nn.l2_loss(model.b1) +
                LAMBDA*tf.nn.l2_loss(model.b2) +
                LAMBDA*tf.nn.l2_loss(model.b3)))

        # Auto Differentiation Takes Place Here
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(zip(grads, model.variables))

        bar.set_description(f"Loss @ {i} => {loss.numpy():0.6f}")
        bar.refresh()

    # Use Meshgrid to find classification of all points on graph
    xv,yv = np.meshgrid(np.linspace(-15,15,500), np.linspace(-15,15,500))
    x_bound = np.reshape(xv.flatten(), (250000, 1))
    y_bound = np.reshape(yv.flatten(), (250000, 1))
    bound = tf.nn.sigmoid(model(x_bound, y_bound))

    figure1 = plt.figure(1)
    # Contour Plot with ColorMap to show Binary Classification
    cs = plt.contourf(xv, yv, np.reshape(bound,(500,500)), [0,.5,1], colors=['violet', 'lime'])
    # Plotting of Both Spirals to Show Accuracy of Classification
    plt.plot(spiral.x1, spiral.y1, 'b.', label='Spiral 1')
    plt.plot(spiral.x2, spiral.y2, 'r.', label='Spiral 2')
    plt.title("Spiral")
    plt.xlabel('x')
    plt.ylabel('y', rotation=0) # Rotate Y-Axis Label
    plt.legend()
    plt.show()
    figure1.savefig('fig1_Elu5.png')
