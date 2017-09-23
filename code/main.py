import numpy as np
from init import loadBias, loadWeight, softmax, D_relu, exportFile, relu

np.set_printoptions(precision=16)


# import matplotlib.pyplot as plt

# Define softmax layer class
class softMaxLayer():
    def __init__(self, w, b):
        self.weight = w
        self.bias = b
    # cross entropy cost function on softmax function
    def cost(self, Y, T):
        return - np.sum(np.multiply(T, np.log(Y))) / Y.shape[0]
    # output of softmax function
    def get_output(self, X):
        return softmax(X)
    def get_input_grad(self, Y, T):
        return (Y - T) / Y.shape[0]

# Define relu layer class
class reluLayer():
    def __init__(self, w, b):
        self.weight = w
        self.bias = b

    def get_output(self, X):
        return relu(X)

    def get_input_grad(self, Y, output_grad):
        return np.multiply(D_relu(Y), output_grad)


def forward(input, layers):
    outputs = [input];
    index = 0
    x = input
    for layer in layers:
        x = layer.get_output(np.dot(np.array(x), layer.weight) + layer.bias)
        outputs.append(x)
        x = outputs[-1]
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
        w_grad = np.float32(X.T.dot(input_grad))
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
            layers.append(reluLayer(w[i],b[i]))
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

def training(layers,x,y,wfile,bfile):
    # forward
    output = forward(x, layers)
    # backpropagation
    w_grads, b_grads = backpropagation(output, y, layers)
    exportFile(bfile, b_grads)
    exportFile(wfile, w_grads)


def main():
    # y_train_raw = np.genfromtxt('../data/y_train.csv', delimiter=",").astype(np.int64)
    # y_train = np.eye(4)[y_train_raw].astype(np.int64)
    # x_train = np.genfromtxt('../data/x_train.csv', delimiter=",").astype(np.int64)
    x_test = np.array([[-1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1]])
    y_test = np.array([[0, 0, 0, 1]])

    # load weight and bias for first NN
    b_100_40_4 = loadBias('../data/b-100-40-4.csv')
    w_100_40_4 = loadWeight('../data/w-100-40-4.csv', [14, 100, 40, 4])
    layer1 = initLayer(1,w_100_40_4,b_100_40_4)

    # load weight and bias for second NN
    b_28_6_4 = loadBias('../data/b-28-6-4.csv')
    w_28_6_4 = loadWeight('../data/w-28-6-4.csv', dimInList(6, 28))
    layer2 = initLayer(2, w_28_6_4, b_28_6_4)

    # load weight and bias for third NN
    b_14_28_4 = loadBias('../data/b-14-28-4.csv')
    w_14_28_4 = loadWeight('../data/w-14-28-4.csv', dimInList(28, 14))
    layer3 = initLayer(3,w_14_28_4,b_14_28_4)

    #run one-time training
    training(layer1,x_test,y_test,'dw-100-40-4.csv','db-100-40-4.csv')
    training(layer2, x_test, y_test, 'dw-28-6-4.csv', 'db-28-6-4.csv')
    training(layer3, x_test, y_test, 'dw-14-28-4.csv', 'db-14-28-4.csv')
    print 'Result generated'


def test():
    x = np.array([[1, 2, 3]])
    w_100_40_4 = loadWeight('../data/w-100-40-4.csv', [14, 100, 40, 4])
    print np.around([0.37876876876876876786876868767, 1.64], decimals=16)


if __name__ == "__main__":
    main()
