import numpy as np
import math
import matplotlib.pyplot as plt
import os.path
from utils import *

np.set_printoptions(precision=16)

# Define softmax layer class
class softmax_layer():
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
class relu_layer():
    def __init__(self, w, b):
        self.weight = w
        self.bias = b

    def get_output(self, X):
        return relu(X)

    def get_input_grad(self, Y, output_grad):
        return np.multiply(relu_derivative(Y), output_grad)


# feedforward process
def forward(input, layers):
    outputs = [input];
    index = 0
    x = input
    for layer in layers:
        x = layer.get_output(np.dot(np.array(x), layer.weight) + layer.bias)
        outputs.append(x)
        index += 1
    return outputs


# backpropagation process
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
def init_layer(type, w, b):
    layers = []
    if type == 1:
        # 2 hidden layers
        layers.append(relu_layer(w[0], b[0]))
        layers.append(relu_layer(w[1], b[1]))
        layers.append(softmax_layer(w[2], b[2]))
    elif type == 2:
        # 6 * 28 hidden layer
        for i in range(6):
            layers.append(relu_layer(w[i], b[i]))
        layers.append(softmax_layer(w[6], b[6]))
    elif type == 3:
        # 28 * 14 hidden layer
        for i in range(28):
            layers.append(relu_layer(w[i], b[i]))
        layers.append(softmax_layer(w[28], b[28]))
    return layers


# one-time training
def training(layers, x, y, wfile, bfile, isTraining, rate):
    # forward
    output = forward(x, layers)

    # backpropagation
    w_grads, b_grads = backpropagation(output, y, layers)
    if isTraining:
        index = 0
        for layer in reversed(layers):
            layer.weight -= w_grads[index] * rate
            layer.bias -= b_grads[index] * rate
            index += 1
    else:
        export_file(bfile, b_grads)
        export_file(wfile, w_grads)


def q4():
    x_test = np.array([[-1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1]])
    y_test = np.array([[0, 0, 0, 1]])

    # load weight and bias for first NN
    b_100_40_4 = load_bias('../c/b-100-40-4.csv')
    w_100_40_4 = load_weight('../c/w-100-40-4.csv', nodes_in_list(1))
    layer1 = init_layer(1, w_100_40_4, b_100_40_4)

    # load weight and bias for second NN
    b_28_6_4 = load_bias('../c/b-28-6-4.csv')
    w_28_6_4 = load_weight('../c/w-28-6-4.csv', nodes_in_list(2))
    layer2 = init_layer(2, w_28_6_4, b_28_6_4)

    # load weight and bias for third NN
    b_14_28_4 = load_bias('../c/b-14-28-4.csv')
    w_14_28_4 = load_weight('../c/w-14-28-4.csv', nodes_in_list(3))
    layer3 = init_layer(3, w_14_28_4, b_14_28_4)

    # run one-time training
    training(layer1, x_test, y_test, 'dw-100-40-4.csv', 'db-100-40-4.csv', False, 1)
    training(layer2, x_test, y_test, 'dw-28-6-4.csv', 'db-28-6-4.csv', False, 1)
    training(layer3, x_test, y_test, 'dw-14-28-4.csv', 'db-14-28-4.csv', False, 1)
    print 'Result generated'


def generate_param(list):
    bias = []
    weight = []
    for i in range(len(list) - 1):
        # w is normal distribution, b is zero
        r = math.sqrt(2.0 / list[i])
        weight.append(np.random.uniform(size=(list[i], list[i + 1]), low=-r, high=r))
        bias.append(np.zeros(shape=(1, list[i + 1])))
    return weight, bias


def accuracy_cost(input, layers, target):
    output = forward(input, layers)
    cost = layers[-1].cost(output[-1], target)
    output_digit = np.zeros_like(output[-1])
    output_digit[np.arange(len(output[-1])), output[-1].argmax(1)] = 1
    tf = np.equal(output_digit, target).all(axis=1)
    return np.float32(np.sum(tf)) / np.float32(target.shape[0]) * 100, cost


def q123(type, step, rate, iter):
    # initial weight and bias for NN and list of layers
    if type == 1:
        nn = '<NN-14-100-40-4>'
        w_100_40_4, b_100_40_4 = generate_param(nodes_in_list(1))
        layers = init_layer(1, w_100_40_4, b_100_40_4)
    elif type == 2:
        nn = '<NN-14-28*6-4>'
        w_28_6_4, b_28_6_4 = generate_param(nodes_in_list(2))
        layers = init_layer(2, w_28_6_4, b_28_6_4)
    elif type == 3:
        nn = '<NN-14-14*28-4>'
        w_14_28_4, b_14_28_4 = generate_param(nodes_in_list(3))
        layers = init_layer(3, w_14_28_4, b_14_28_4)
    else:
        print "wrong NN numer"
        return

    figName = 'NN' + str(type) + '-' + str(step) + '-' + str(rate) + '-' + str(iter) + '-train-cost.png'
    figName2 = 'NN' + str(type) + '-' + str(step) + '-' + str(rate) + '-' + str(iter) + '-train-accuracy.png'
    figName3 = 'NN' + str(type) + '-' + str(step) + '-' + str(rate) + '-' + str(iter) + '-test-cost.png'
    figName4 = 'NN' + str(type) + '-' + str(step) + '-' + str(rate) + '-' + str(iter) + '-test-accuracy.png'
    scores_train, costs_train, scores_test, costs_test = stochastic_gradient_descend(step, rate, layers, iter)

    # draw figure of cost data
    plt.figure(0)
    plt.plot(costs_train)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.suptitle('Cross Entropy Cost on train data set ' + nn)
    plt.title('The minimum cost is ' + str(np.amin(costs_train)))
    plt.grid(True)
    if os.path.isfile(figName):
        os.remove(figName)
    plt.savefig(figName)

    # draw figure of accuracy data
    plt.figure(1)
    plt.plot(scores_train)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.suptitle('Accuracy in percentage on train data set ' + nn)
    plt.title('The maximum accuracy is ' + str(np.around(np.amax(scores_train), decimals=2)) + '%')
    plt.grid(True)
    if os.path.isfile(figName2):
        os.remove(figName2)
    plt.savefig(figName2)

    # draw figure of cost data
    plt.figure(2)
    plt.plot(costs_test)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.suptitle('Cross Entropy Cost on test data set ' + nn)
    plt.title('The minimum cost is ' + str(np.amin(costs_test)))
    plt.grid(True)
    if os.path.isfile(figName3):
        os.remove(figName3)
    plt.savefig(figName3)

    # draw figure of accuracy data
    plt.figure(3)
    plt.plot(scores_test)
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.suptitle('Accuracy in percentage on test data set ' + nn)
    plt.title('The maximum accuracy is ' + str(np.around(np.amax(scores_test), decimals=2)) + '%')
    plt.grid(True)
    if os.path.isfile(figName4):
        os.remove(figName4)
    plt.savefig(figName4)


def stochastic_gradient_descend(step, learning_rate, layers, iter):
    x_train, x_test, y_train, y_test = load_train_and_test_data()
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
        print 'iteration % d, the accuracy on train set is %.2f , and cost is %.2f' % (j + 1, score_train, cost_train)
        # print 'iteration %d, the accuracy on test set is %.2f , and cost is %.2f' % (j + 1, score_test, cost_test)
        scores_test.append(score_test)
        costs_test.append(cost_test)
        scores_train.append(score_train)
        costs_train.append(cost_train)
    return scores_train, costs_train, scores_test, costs_test


if __name__ == "__main__":
    # Draw cost and accuracy plot for test and train data
    # param: NN type, batch size, learning rate, iteration
    q123(1,32,0.09,50)

    # Output dw and db files
    q4()
