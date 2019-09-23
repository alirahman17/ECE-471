#!/bin/python3.6

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import trange

NUM_FEATURES = 10 # This is really M in this code
NUM_SAMP = 50
BATCH_SIZE = 32
NUM_BATCHES = 300
LEARNING_RATE = 0.1

class Data(object):
    def __init__(self, num_features=NUM_FEATURES, num_samp=NUM_SAMP):
        """
        Draw random weights and bias. Project vectors in R^NUM_FEATURES
        onto R with said weights and bias.
        """
        num_samp = NUM_SAMP
        sigma = 0.1
        np.random.seed(31415)

        self.index = np.arange(num_samp)
        self.x = np.sort(np.random.uniform(size=(num_samp, 1)), axis=0)
        self.E = np.random.normal(scale=sigma, size=(num_samp, 1))
        self.y = np.sin(2 * np.pi * self.x) + self.E
        self.sine = np.sin(2 * np.pi * self.x)

    def get_batch(self, batch_size=BATCH_SIZE):
        """
        Select random subset of examples for training batch
        """
        choices = np.random.choice(self.index, size=batch_size)
        return self.x[choices].flatten(), self.y[choices].flatten()


class Model(tf.Module):
    def __init__(self, num_features=NUM_FEATURES):
        self.w = tf.Variable(tf.random.uniform(shape=[num_features, 1]))
        self.b = tf.Variable(tf.zeros(shape=[1, 1]))
        self.mu = tf.Variable(tf.random.uniform(shape=[num_features, 1]))
        self.sig = tf.Variable(tf.random.uniform(shape=[num_features, 1]))

    def phi(self, x, mu, sigma):
        return tf.exp(-tf.square((x-mu)/sigma))

    def __call__(self, x):
        return tf.squeeze(tf.matmul(tf.transpose(self.w),self.phi(x, self.mu, self.sig)) + self.b)


if __name__ == "__main__":
    data = Data()
    model = Model()
    optimizer = tf.optimizers.SGD(learning_rate=LEARNING_RATE)

    bar = trange(NUM_BATCHES)
    for i in bar:
        with tf.GradientTape() as tape:
            x, y = data.get_batch()
            y_hat = model(x)
            loss = tf.reduce_mean((y_hat - y) ** 2)

        # Auto Differentiation Occurs Here
        grads = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(zip(grads, model.variables))

        bar.set_description(f"Loss @ {i} => {loss.numpy():0.6f}")
        bar.refresh()

    w_hat = np.squeeze(model.w.numpy())
    b_hat = np.squeeze(model.b.numpy())
    sig_hat = np.squeeze(model.sig.numpy())
    mu_hat = np.squeeze(model.mu.numpy())

    # print out estimates
    print("b_hat")
    print(f"{b_hat:0.2f}")

    print("mu_hat")
    for a in mu_hat:
        print(f"{a:0.2f}")

    print("sigma_hat")
    for a in sig_hat:
        print(f"{a:0.2f}")

    print("w_hat")
    for a in w_hat:
        print(f"{a:0.2f}")

    figure1 = plt.figure(1)
    plt.plot(data.x ,data.y, 'go')
    plt.plot(data.x, data.sine)
    plt.plot(data.x, model(data.x.flatten()), color="red", linestyle='dotted')
    plt.title("Sine Waves")
    plt.xlabel('x')
    plt.ylabel('y')

    figure2 = plt.figure(2)
    for phi in range(NUM_FEATURES):
        x_phi = np.linspace(-.5, 1.5, 1000)
        bases = tf.exp(-tf.square((x_phi.flatten() - mu_hat[phi]) / sig_hat[phi]))
        plt.plot(x_phi, bases)

    plt.ylim(0,1.1)
    plt.xlim(-.5,1.5)
    plt.title("Bases")
    plt.xlabel('x')
    plt.ylabel('y')

    figure1.savefig('fig1.png', bbox_inches= 'tight')
    figure2.savefig('fig2.png', bbox_inches= 'tight')

    plt.show()
