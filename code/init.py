import numpy as np
import csv


# load bias
def loadBias(array, file):
    with open(file) as f:
        reader = csv.reader(f)
        for row in reader:
            array.append(np.array(row[1:]).astype(float))

def loadWeight(array,file,nodes):
    with open(file) as f:
        reader = csv.reader(f)
        layer=0
        index=0
        rowData = np.empty(shape=(nodes[layer], nodes[layer+1]))
        for row in reader:
            rowData[index] = np.array(row[1:]).astype(float)
            index+=1
            if(index == nodes[layer]):
                array.append(rowData)
                layer+=1
                index =0
                if(layer==3):
                    return
                rowData=np.empty(shape=(nodes[layer], nodes[layer+1]))

# define activation function

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)


def relu(z):
    return np.maximum(z, 0, z)


class layer(object):
    def get_iter(self):
        return []

    def get_grad(self, X, output_grad):
        return []

    def get_ouput(self, X):
        pass

    def get_input_grad(self, Y, output_grad=None, T=None):
        pass


class relu(layer):
    def __init__(self, n_in, n_out):
        self.W = [];
        self.b = [];
