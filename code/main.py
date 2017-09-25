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


def result(input, layers):
    output = forward(input, layers)
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
        w_grad = np.float32(X.T.dot(input_grad) / input_grad.shape[0])
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
def dimInList(type):
    dim = [14]
    if type==1:
        dim.append(100)
        dim.append(40)
    elif type==2:
        for i in range(6):
            dim.append(28)
    elif type==3:
        for i in range(28):
            dim.append(14)
    dim.append(4)
    return dim


def training(layers, x, y, wfile, bfile, isTraining, rate):
    # forward
    output = forward(x, layers)

    # save copy of output
    output_copy = copy.deepcopy(output)

    # backpropagation
    w_grads, b_grads = backpropagation(output_copy, y, layers)
    if isTraining:
        index = 0
        for layer in reversed(layers):
            layer.weight -= w_grads[index] * rate
            layer.bias -= b_grads[index] * rate
            index += 1
    else:
        exportFile(bfile, b_grads)
        exportFile(wfile, w_grads)


def q4():
    x_test = np.array([[-1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1]])
    y_test = np.array([[0, 0, 0, 1]])

    # load weight and bias for first NN
    b_100_40_4 = loadBias('../b/b-100-40-4.csv')
    w_100_40_4 = loadWeight('../b/w-100-40-4.csv', dimInList(1))
    layer1 = initLayer(1, w_100_40_4, b_100_40_4)

    # load weight and bias for second NN
    b_28_6_4 = loadBias('../b/b-28-6-4.csv')
    w_28_6_4 = loadWeight('../b/w-28-6-4.csv', dimInList(2))
    layer2 = initLayer(2, w_28_6_4, b_28_6_4)

    # load weight and bias for third NN
    b_14_28_4 = loadBias('../b/b-14-28-4.csv')
    w_14_28_4 = loadWeight('../b/w-14-28-4.csv', dimInList(3))
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
        r = math.sqrt(2.0 / 14)
        weight.append(np.random.uniform(size=(14, 100), low=-r, high=r))
        r = math.sqrt(2.0 / 100)
        weight.append(np.random.uniform(size=(100, 40), low=-r, high=r))
        r = math.sqrt(2.0 / 40)
        weight.append(np.random.uniform(size=(40, 4), low=-r, high=r))
        return weight, [np.zeros(shape=(1, 100)), np.zeros(shape=(1, 40)), np.zeros(shape=(1, 4))]
    elif type == 2:
        # bias
        for i in range(6):
            bias.append(np.zeros(shape=(1, 28)))
        bias.append(np.zeros(shape=(1, 4)))
        # weight
        r = math.sqrt(2.0 / 14)
        weight.append(np.random.uniform(size=(14, 28), low=-r, high=r))
        for i in range(5):
            r = math.sqrt(2.0 / 28)
            weight.append(np.random.uniform(size=(28, 28), low=-r, high=r))
        r = math.sqrt(2.0 / 28)
        weight.append(np.random.uniform(size=(28, 4), low=-r, high=r))
        return weight, bias
    elif type == 3:
        for i in range(28):
            bias.append(np.ones(shape=(1, 14)))
            weight.append(np.random.randn(14, 14))
        bias.append(np.ones(shape=(1, 4)))
        weight.append(np.random.randn(14, 4))
        return weight, bias


def generateParam(list):
    bias = []
    weight = []
    for i in range(len(list)-1):
        r = math.sqrt(2.0 / list[i])
        weight.append(np.random.uniform(size=(list[i], list[i+1]), low=-r, high=r))
        bias.append(np.zeros(shape=(1, list[i+1])))
    return weight,bias


def loadinitW(path, loop):
    w = []
    for i in range(loop):
        w.append(np.loadtxt(path + '/w' + str(i)))
    return w


def storeParameter(path, loop, w):
    shutil.rmtree(path)
    os.makedirs(path)
    for i in range(loop):
        np.savetxt(path + '/w' + str(i), w[i])


def accuracy_cost(input, layers, target):
    output = forward(input, layers)
    cost = layers[-1].cost(output[-1], target)
    output_digit = np.zeros_like(output[-1])
    output_digit[np.arange(len(output[-1])), output[-1].argmax(1)] = 1
    tf = np.equal(output_digit, target).all(axis=1)
    return np.float32(np.sum(tf)) / np.float32(target.shape[0]) * 100, cost


def q123():
    # initial weight and bias for first NN
    w_100_40_4, b_100_40_4 = generateParam(dimInList(1))
    layer1 = initLayer(1, w_100_40_4, b_100_40_4)
    # initial weight and bias for second NN
    w_28_6_4, b_28_6_4 = generateParam(dimInList(2))
    layer2 = initLayer(2, w_28_6_4, b_28_6_4)
    # initial weight and bias for third NN
    w_14_28_4, b_14_28_4 = generateParam(dimInList(3))
    layer3 = initLayer(3, w_14_28_4, b_14_28_4)

    #oneScores_train, oneCosts_train, oneScores_test,oneCosts_test = stochastic_gradient_descend(50, 0.1, layer1, 50)
    #twoScores_train, twoCosts_train, twoScores_test, twoCosts_test = stochastic_gradient_descend(50, 0.1, layer2, 100)
    threeScores_train, threeCosts_train, threeScores_test, threeCosts_test = stochastic_gradient_descend(32, 1, layer3, 100)

    #line1 = plt.plot(oneCosts_train)
    #plt.setp(line1, color='b', linewidth=1.0)

    # line2 = plt.plot(twoCosts_train)
    # plt.setp(line2, color='g', linewidth=1.0)
    #
    line3 = plt.plot(threeCosts_train)
    plt.setp(line3, color='r', linewidth=1.0)

    plt.ylabel('Cross Entropy Cost')
    plt.show()
    raw_input()



def stochastic_gradient_descend(step, learning_rate, layers, iter):
    x_train, x_test, y_train, y_test = loadTraningTestData()
    costs_test = []
    scores_test = []
    costs_train = []
    scores_train = []
    for j in range(iter):
        x_train, y_train = unison_shuffled_copies(x_train, y_train)
        for i in range(0, x_train.shape[0], step):
            training(layers, x_train[i:i + step], y_train[i:i + step], '', '', True, learning_rate)
        score_train, cost_train = accuracy_cost(x_train, layers, y_train)
        score_test, cost_test = accuracy_cost(x_test, layers, y_test)
        print 'iteration % d, the accuracy on train set is %.2f , and cost is %.2f' %(j+1,score_train,cost_train)
        print 'iteration %d, the accuracy on test set is %.2f , and cost is %.2f' % (j + 1, score_test, cost_test)
        scores_test.append(score_test)
        costs_test.append(cost_test)
        scores_train.append(score_train)
        costs_train.append(cost_train)
    # print 'NN is trained, the accuracy on train set is %.2f , the ' % np.amax(scores)
    return scores_train, costs_train,scores_test,costs_test


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def test():
    return


if __name__ == "__main__":
    q123()
