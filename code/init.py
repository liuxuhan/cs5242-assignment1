import numpy as np
import csv
import os.path


# load bias
def loadBias(file):
    array = []
    with open(file) as f:
        reader = csv.reader(f)
        for row in reader:
            array.append(np.array(row[1:]).astype(float))
    return array


def loadWeight(file, nodes):
    array = []
    stopPoint = len(nodes) -1
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
    if np.amax(z) >= 10000:
        z = z - np.amax(z)
    z = np.float128(z)
    return np.exp(z) / np.sum(np.exp(z))


def relu(z):
    return np.maximum(z, 0, z)


def D_relu(X):
    X[X==0] = -1
    return 1 * (X > 0)


def exportFile(fileName, data):
    addr = '../e0146241-test/' + fileName
    if os.path.isfile(addr):
        os.remove(addr)

    with open(addr, 'ab') as fp:
        for item in reversed(data):
            np.savetxt(fp, item, delimiter=',', fmt='%.16g')
