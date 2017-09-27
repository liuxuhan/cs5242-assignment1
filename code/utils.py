import numpy as np
import csv
import os.path
import copy
import shutil


# load bias
def load_bias(file):
    array = []
    with open(file) as f:
        reader = csv.reader(f)
        for row in reader:
            array.append(np.array(row[1:]).astype(float))
    return array


def load_weight(file, nodes):
    array = []
    stopPoint = len(nodes) - 1
    with open(file) as f:
        reader = csv.reader(f)
        layer = 0
        index = 0
        rowData = np.empty(shape=(nodes[layer], nodes[layer + 1]))
        for row in reader:
            rowData[index] = np.array(row[1:])
            index += 1
            if (index == nodes[layer]):
                array.append(rowData)
                layer += 1
                index = 0
                if (layer == stopPoint):
                    return array
                rowData = np.empty(shape=(nodes[layer], nodes[layer + 1]))


# define activation function

def softmax(z):
    z = z - np.amax(z)
    exps = np.exp(z)
    return exps / np.sum(exps, axis=1, keepdims=True)


def relu(z):
    return np.maximum(z, 0, z)


def relu_derivative(X):
    return 1 * (X > 0)


def export_file(fileName, data):
    addr = '../e0146241/' + fileName
    if os.path.isfile(addr):
        os.remove(addr)

    with open(addr, 'ab') as fp:
        for item in reversed(data):
            np.savetxt(fp, item, delimiter=',', fmt='%.17g')


# list of number of node in each layers
def nodes_in_list(type):
    nodes = [14]
    if type == 1:
        nodes.append(100)
        nodes.append(40)
    elif type == 2:
        for i in range(6):
            nodes.append(28)
    elif type == 3:
        for i in range(28):
            nodes.append(14)
    nodes.append(4)
    return nodes


# load data for q123
def load_train_and_test_data():
    # training data
    x_train = np.genfromtxt('../data/x_train.csv', delimiter=",").astype(np.float32)
    # translate class format into 0/1 format
    y_train_raw = np.genfromtxt('../data/y_train.csv', delimiter=",").astype(np.int64)
    y_train = np.eye(4)[y_train_raw].astype(np.float32)

    # test data
    x_test = np.genfromtxt('../data/x_test.csv', delimiter=",").astype(np.float32)
    # translate class format into 0/1 format
    y_test_raw = np.genfromtxt('../data/y_test.csv', delimiter=",").astype(np.int64)
    y_test = np.eye(4)[y_test_raw].astype(np.float32)

    return x_train, x_test, y_train, y_test


# shuffle whole dataset x and y
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
