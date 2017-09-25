import numpy as np
import math
import copy
import matplotlib.pyplot as plt
import shutil
import os.path
from init import loadBias, loadWeight, softmax, D_relu, exportFile, relu


np.set_printoptions(precision=16)


# Define softmax layer class
class softMaxLayer():
    def __init__(self, w, b):
        self.weight = w
        self.bias = b

    # cross entropy cost function on softmax function
    def cost(self, Y, T):
        # avoid invalid value
        Y[Y == 0] = math.e
        return - np.sum(np.multiply(T, np.log(Y))) / Y.shape[0]

    # output of softmax function
    def get_output(self, X):
        return softmax(X)

    def get_input_grad(self, Y, T):
        return (Y - T)


# Define relu layer class
class reluLayer():
    def __init__(self, w, b):
        self.weight = w
        self.bias = b

    def get_output(self, X):
        return relu(X)

    def get_input_grad(self, Y, output_grad):
        return np.multiply(D_relu(Y), output_grad)

def result(input ,layers):
    output = forward(input,layers)
    return np.argmax(output[-1])


def forward(input, layers):
    outputs = [input];
    index = 0
    x = input
    for layer in layers:
        x = layer.get_output(np.dot(np.array(x), layer.weight) + layer.bias)
        outputs.append(x)
        index += 1
    return outputs


def backpropagation(activations, targets, layers):
    output_grad = None
    b_grads = []
    w_grads = []
    for layer in reversed(layers):
        Y = activations.pop()
        if output_grad is None:
            input_grad = layer.get_input_grad(Y, targets)
        else:
            input_grad = layer.get_input_grad(Y, output_grad)
        X = activations[-1]
        w_grad = np.float32(X.T.dot(input_grad)/input_grad.shape[0])
        if input_grad.shape[0] > 1:
            b_grad = np.float32(np.mean(input_grad, axis=0))
        else:
            b_grad = np.float32(input_grad)
        w_grads.append(w_grad)
        b_grads.append(b_grad)
        output_grad = input_grad.dot(layer.weight.T)
    return w_grads, b_grads


# Create a list of layer class
def initLayer(type, w, b):
    layers = []
    if type == 1:
        # 2 hidden layers
        layers.append(reluLayer(w[0], b[0]))
        layers.append(reluLayer(w[1], b[1]))
        layers.append(softMaxLayer(w[2], b[2]))
    elif type == 2:
        # 6 * 28 hidden layer
        for i in range(6):
            layers.append(reluLayer(w[i], b[i]))
        layers.append(softMaxLayer(w[6], b[6]))
    elif type == 3:
        # 28 * 14 hidden layer
        for i in range(28):
            layers.append(reluLayer(w[i], b[i]))
        layers.append(softMaxLayer(w[28], b[28]))
    return layers


# list of dimension
def dimInList(layers, nodes):
    dim = [14]
    for i in range(layers):
        dim.append(nodes)
    dim.append(4)
    return dim


def training(layers, x, y, wfile, bfile, isTraining, rate, costList):
    # forward
    output = forward(x, layers)

    # save copy of output
    output_copy = copy.deepcopy(output)

    #calculate cost
    costList.append(layers[-1].cost(output[-1], y))

    # backpropagation
    w_grads, b_grads = backpropagation(output_copy, y, layers)
    if isTraining:
        index = 0
        for layer in reversed(layers):
            layer.weight -= w_grads[index] * rate
            layer.bias -= b_grads[index]
            index += 1
    else:
        exportFile(bfile, b_grads)
        exportFile(wfile, w_grads)


def q4():
    x_test = np.array([[-1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1]])
    y_test = np.array([[0, 0, 0, 1]])

    # load weight and bias for first NN
    b_100_40_4 = loadBias('../b/b-100-40-4.csv')
    w_100_40_4 = loadWeight('../b/w-100-40-4.csv', [14, 100, 40, 4])
    layer1 = initLayer(1, w_100_40_4, b_100_40_4)

    # load weight and bias for second NN
    b_28_6_4 = loadBias('../b/b-28-6-4.csv')
    w_28_6_4 = loadWeight('../b/w-28-6-4.csv', dimInList(6, 28))
    layer2 = initLayer(2, w_28_6_4, b_28_6_4)

    # load weight and bias for third NN
    b_14_28_4 = loadBias('../b/b-14-28-4.csv')
    w_14_28_4 = loadWeight('../b/w-14-28-4.csv', dimInList(28, 14))
    layer3 = initLayer(3, w_14_28_4, b_14_28_4)

    # run one-time training
    training(layer1, x_test, y_test, 'dw-100-40-4.csv', 'db-100-40-4.csv', False, 1, [])
    training(layer2, x_test, y_test, 'dw-28-6-4.csv', 'db-28-6-4.csv', False, 1, [])
    training(layer3, x_test, y_test, 'dw-14-28-4.csv', 'db-14-28-4.csv', False, 1, [])
    print 'Result generated'


def loadTraningTestData():
    y_train_raw = np.genfromtxt('../data/y_train.csv', delimiter=",").astype(np.int64)
    y_train = np.eye(4)[y_train_raw].astype(np.float32)
    x_train = np.genfromtxt('../data/x_train.csv', delimiter=",").astype(np.float32)
    x_test = np.genfromtxt('../data/x_test.csv', delimiter=",").astype(np.float32)
    y_test_raw = np.genfromtxt('../data/y_test.csv', delimiter=",").astype(np.int64)
    y_test = np.eye(4)[y_test_raw].astype(np.float32)
    return x_train, x_test, y_train, y_test


def generateParameter(type):
    bias = []
    weight = []
    if type == 1:
        weight.append(np.random.uniform(14, 100))
        weight.append(np.random.uniform(100, 40))
        weight.append(np.random.uniform(40, 4))
        return weight, [np.ones(shape=(1, 100)), np.ones(shape=(1, 40)), np.ones(shape=(1, 4))]
    elif type == 2:
        for i in range(6):
            bias.append(np.zeros(shape=(1, 28)))
            bias.append(np.zeros(shape=(1, 4)))

        r = math.sqrt(2.0/14)
        weight.append(np.random.uniform(size=(14, 28),low=-r,high=r))
        for i in range(5):
            r = math.sqrt(2.0/28)
            weight.append(np.random.uniform(size=(28, 28),low=-r,high=r))
        r = math.sqrt(2.0/28)
        weight.append(np.random.uniform(size=(28, 4),low=-r,high=r))
        return weight, bias
    elif type == 3:
        for i in range(28):
            bias.append(np.ones(shape=(1, 14)))
            weight.append(np.random.randn(14, 14))
        bias.append(np.ones(shape=(1, 4)))
        weight.append(np.random.randn(14, 4))
        return weight, bias

def loadinitW(path,loop):
    w = []
    for i in range(loop):
        w.append(np.loadtxt(path+'/w'+str(i)))
    return w

def storeParameter(path,loop,w):
    shutil.rmtree(path)
    os.makedirs(path)
    for i in range(loop):
        np.savetxt(path + '/w' + str(i), w[i])

def accuracy(input,layers,target):
    output = forward(input,layers)
    output_last_layer = output[len(layers)]
    output_digit = np.zeros_like(output_last_layer)
    output_digit[np.arange(len(output_last_layer)), output_last_layer.argmax(1)] = 1
    tf = np.equal(output_digit, target).all(axis=1)
    return np.float32(np.sum(tf))/np.float32(target.shape[0])*100

def q123():
    x_train, x_test, y_train, y_test = loadTraningTestData()

    # initial weight and bias for first NN
    w_100_40_4, b_100_40_4 = generateParameter(1)
    # w_100_40_4 = loadinitW('../initParam1',3)
    layer1 = initLayer(1, w_100_40_4, b_100_40_4)
    #storeParameter('../initParam1',3,w_100_40_4)


    #initial weight and bias for second NN
    w_28_6_4, b_28_6_4 = generateParameter(2)
    layer2 = initLayer(2, w_28_6_4, b_28_6_4)
    #
    # # initial weight and bias for third NN
    # w_14_28_4,b_14_28_4 = generateParameter(3)
    # layer3 = initLayer(3, w_14_28_4, b_14_28_4)

    cost = []
    score = []
    step = 10
    learning_rate = 1
    usedLayer = layer2
    for i in range(0,x_train.shape[0],step):
        training(usedLayer, x_train[i:i+step], y_train[i:i+step], '', '', True, learning_rate, cost)
        score.append(accuracy(x_train,usedLayer,y_train))

    print 'NN is trained'
    print np.amax(score)
    line = plt.plot(cost)
    plt.setp(line, color='b', linewidth=1.0)
    plt.ylabel('Cross Entropy Cost')
    plt.show()

    # for i in range(len(x_test)):
    #     training(layer1, np.array([x_test[i]]), np.array([y_test[i]]), '', '', True, 0.5, cost_test)
    # print 'NN1 is tested'
    #
    # for i in range(len(x_train)):
    #     training(layer2, np.array([x_train[i]]), np.array([y_train[i]]), '', '', True, 0.5, cost2)
    # print 'NN2 is trained'
    #
    # for i in range(len(x_train)):
    #     training(layer3, np.array([x_train[i]]), np.array([y_train[i]]), '', '', True, 0.5, cost3)
    # print 'NN3 is trained'

    #print 'NN1 return result %d' % result(np.array([[-1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1]]),layer1)
    # print 'NN2 return result %d' % result(np.array([[-1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1]]), layer2)
    # print 'NN3 return result %d' % result(np.array([[-1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1]]), layer3)

    # line1 = plt.plot(cost1)
    # plt.setp(line1, color='b', linewidth=1.0)
    # plt.ylabel('Cross Entropy Cost')
    # plt.show()



def test():
    print 'sds'


if __name__ == "__main__":
    q4()