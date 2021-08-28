#!/usr/bin/env python
'''
H Sun, Waseda Univ., 2021
'''

import numpy as np
import matplotlib.pyplot as plt
from random import shuffle


def sigmoid(x):
    # activation function: sigmoid
    return 1 / (1 + np.exp(-x))


def dsigmoid(x):
    # derivative of sigmoid
    return sigmoid(x) * (1 - sigmoid(x))


def crossEntropy(l, y):
    # loss function: cross entropy
    return -l * np.log(y) - (1 - l) * np.log(1 - y)


def dcrossEntropy(l, y):
    # derivative of cross entropy
    return (1 - l) / (1 - y) - l / y


def gen_sprial(num):
    n = np.linspace(0, 99, num)

    alpha1 = np.pi * (n - 1) / 25
    beta = 0.4 * ((105 - n) / 104)
    x0 = 0.5 + beta * np.sin(alpha1)
    y0 = 0.5 + beta * np.cos(alpha1)
    x1 = 0.5 - beta * np.sin(alpha1)
    y1 = 0.5 - beta * np.cos(alpha1)
    plt.scatter(x0, y0, marker='+')
    plt.scatter(x1, y1, marker='o')
    plt.axis('equal')
    plt.pause(0.1)
    # arrange coordinates in one list

    X = []
    L = []
    for _ in range(0, num):
        X.append([x0[_], y0[_]])
        L.append(0)

    for _ in range(0, num):
        X.append([x1[_], y1[_]])
        L.append(1)
    return X, L


def shuffle_data(X, L, num):
    # shuffle the dataset
    ind = [_ for _ in range(0, 2 * num)]
    shuffle(ind)
    return np.array(X)[ind], np.array(L)[ind]


def split_data(X, L, ratio):
    size_train = np.int(np.floor(len(X) * ratio))
    Xtrain = X[:size_train]
    Ltrain = L[:size_train]
    Xtest = X[size_train:]
    Ltest = L[size_train:]
    return Xtrain, Ltrain, Xtest, Ltest


def gen_data(ratio, num):
    X, L = gen_sprial(num)
    X, L = shuffle_data(X, L, num)
    Xtrain, Ltrain, Xtest, Ltest = split_data(X, L, ratio)
    return Xtrain, Ltrain, Xtest, Ltest


def gen_coord():
    # generate coordinate in area [0,0] to [1,1]
    r = np.linspace(0, 1, 100)  # interval of x, y axis
    coord = np.meshgrid(r, r)  # get all coordinates
    X = []
    # arrange coordinates in one list
    for i in range(0, len(coord[0][0])):
        for j in range(len(coord[0]) - 1, -1, -1):
            X.append([coord[0][i][j], coord[1][i][j]])
    X.reverse()
    return X


class Neuron:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def forward(self, x):
        # z = x·w + b
        z = np.dot(x, self.weight) + self.bias
        return z, sigmoid(z)

    def backward(self, error, z):
        # delta = error * f'(z)
        return error * dsigmoid(z)


class ANN:
    def __init__(self, w1, b1, w2, b2, w3, b3, learning_rate):
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.hidden_layer1 = Neuron(self.w1, self.b1)
        self.hidden_layer2 = Neuron(self.w2, self.b2)
        self.output_layer = Neuron(self.w3, self.b3)
        self.learning_rate = learning_rate

    def forward(self, x, l):
        # meaning: z = x·w + b, a = activation_func(z)
        self.z1, self.a1 = self.hidden_layer1.forward(x)
        self.z2, self.a2 = self.hidden_layer2.forward(self.a1)
        self.z3, self.y = self.output_layer.forward(self.a2)
        self.loss = crossEntropy(l, self.y)
        return self.loss, self.y

    def backward(self, x, l):
        error3 = dcrossEntropy(l, self.y)
        delta3 = self.output_layer.backward(error3, self.z3)

        # dw = input_from_last_layer · delta_this_layer
        dw3 = np.dot(self.a2.T, delta3)
        db3 = np.sum(delta3, axis=0)

        error2 = np.dot(delta3, self.w3.T)
        delta2 = self.hidden_layer2.backward(error2, self.z2)

        dw2 = np.dot(self.a1.T, delta2)
        db2 = np.sum(delta2, axis=0)

        error1 = np.dot(delta2, self.w2.T)
        delta1 = self.hidden_layer1.backward(error1, self.z1)

        dw1 = np.dot(np.array([x]).T, delta1)
        db1 = np.sum(delta1, axis=0)

        # update weights and biases
        self.w3 -= self.learning_rate * dw3
        self.b3 -= self.learning_rate * db3
        self.w2 -= self.learning_rate * dw2
        self.b2 -= self.learning_rate * db2
        self.w1 -= self.learning_rate * dw1
        self.b1 -= self.learning_rate * db1


def main():
    #########################################################################################
    # SETUP #################################################################################

    # amount of neurons for each layer
    input_neurons = 2
    hidden_neurons1 = 30
    hidden_neurons2 = 30
    output_neurons = 1

    # generate initial weights and biases
    np.random.seed(3)
    w1 = np.random.randn(input_neurons, hidden_neurons1)
    b1 = -np.random.randn(1, hidden_neurons1)
    w2 = np.random.randn(hidden_neurons1, hidden_neurons2)
    b2 = -np.random.randn(1, hidden_neurons2)
    w3 = np.random.randn(hidden_neurons2, output_neurons)
    b3 = -np.random.randn(1, output_neurons)
    learning_rate = 0.1

    # generate dataset
    n = 200  # n points for each type, size of dataset = 2n
    ratio = 0.8  # ratio = size(training_set)/size(test_set)
    X, L, Xt, Lt = gen_data(ratio, n)  # X, L: training set, Xt, Lt: test set

    network = ANN(w1=w1, b1=b1, w2=w2, b2=b2, w3=w3, b3=b3, learning_rate=learning_rate)
    loss_list = []
    epoch_loss_list = []
    epoch_loss = 0  # to calculate average loss of each epoch
    training_epoch = 1000

    #########################################################################################
    # TRAIN #################################################################################

    for epoch in range(training_epoch):
        for i in range(0, len(X)):
            x = X[i]
            l = L[i]
            loss, _ = network.forward(x, l)  # feed forward
            loss = np.float64(loss)
            print(f"Epoch: {epoch} - {i}, Loss: {loss}")
            network.backward(x, l)  # feed backward
            epoch_loss += loss
            loss_list.append(loss)
        # calculate average loss for each epoch
        epoch_loss_list.append(epoch_loss / (2 * n))
        epoch_loss = 0

    # plot loss
    plt.figure()
    plt.plot([_ for _ in range(0, training_epoch)], epoch_loss_list)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.pause(0.1)

    #########################################################################################
    # TEST ##################################################################################

    hit = 0
    type0 = []
    type1 = []

    for i in range(0, len(Xt)):
        x = Xt[i]
        l = Lt[i]
        _, y = network.forward(x, l)  # get output
        if np.abs(y - l) < 0.5:  # correct output
            hit += 1
        if y < 0.5:
            # predicted type 0, store the input
            type0.append([x.tolist()[0], x.tolist()[1]])
        else:
            # predicted type 1, store the input
            type1.append([x.tolist()[0], x.tolist()[1]])

    #########################################################################################
    # RESULT ################################################################################

    # plot prediction result
    plt.figure()
    plt.scatter(np.mat(type0).T[0].tolist()[0], np.mat(type0).T[1].tolist()[0], marker='+')
    plt.scatter(np.mat(type1).T[0].tolist()[0], np.mat(type1).T[1].tolist()[0], marker='o')
    plt.axis('equal')
    plt.axis([0, 1, 0, 1])
    # plt.pause(0.1)

    # accuracy = hit / size(test_set)
    acc = hit / (len(Xt))
    print("Accuracy:", acc)

    # plot decision boundary using heatmap
    X = gen_coord()
    output = []
    for i in range(0, 10000):
        x = X[i]
        _, y = network.forward(x, 0)  # get output
        output.append(np.float64(y))

    # reshape list into 100*100 array
    z = np.array(output).reshape(100, 100)

    # plt.figure()
    plt.imshow(z, extent=(0, 1, 0, 1))
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    main()
    input()