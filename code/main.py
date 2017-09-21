import numpy as np
from init import loadBias, loadWeight, relu, softmax


# import matplotlib.pyplot as plt




# print len(b_100_40_4)
# print w_100_40_4[0].shape


class softMaxLayer():
    def __init__(self, w, b):
        self.weight = w
        self.bias = b

    def cost(self, Y, T):
        return - np.sum(np.multiply(T, np.log(Y)))

    def get_output(self, X):
        return softmax(X)

    def delta(self, ):
        return


class Layer(object):
    def get_output(self, X):
        pass


class relu(Layer):
    def __init__(self, w, b):
        self.weight = w
        self.bias = b

    def get_output(self, X):
        return np.maximum(X, 0, X)


def forward(input, layers):
    outputs = [input];
    index = 0
    x = input
    for layer in layers:
        x = layer.get_output(np.dot(np.array(x), layer.weight) + layer.bias)
        outputs.append(x)
        x = outputs[-1]
        index+=1
    return outputs


def main():
    y_train_raw = np.genfromtxt('../data/y_train.csv', delimiter=",").astype(np.int64)
    y_train = np.eye(4)[y_train_raw].astype(np.int64)
    x_train = np.genfromtxt('../data/x_train.csv', delimiter=",").astype(np.int64)
    b_100_40_4 = []
    w_100_40_4 = []
    loadBias(b_100_40_4, '../data/b-100-40-4.csv')
    loadWeight(w_100_40_4, '../data/w-100-40-4.csv', [14, 100, 40, 4])
    layer = []
    layer.append(relu(w_100_40_4[0], b_100_40_4[0]))
    layer.append(relu(w_100_40_4[1], b_100_40_4[1]))
    layer.append(softMaxLayer(w_100_40_4[2], b_100_40_4[2]))
    output = forward(x_train[0], layer)
    print output[3]
    print output[3].shape
    soft = softMaxLayer(10,1)
    print soft.cost(output[3],[0,0,0,1])

def test():
    print np.array([[1, 2, 2, 3, 4],[1, 2, 2, 3, 4]]) + np.array([1, 2, 2, 2, 2])


if __name__ == "__main__":
    main()
