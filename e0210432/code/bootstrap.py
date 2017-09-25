from NeuralNetwork import *


if __name__ == "__main__":
    #nn = NeuralNetwork((14,100,40,4), "100-40-4")
    nn = NeuralNetwork((14,28,28,28,28,28,28,4), "28-6-4")
    # nn = NeuralNetwork((14,14,14,14,14,14,14,14,14,
    #                     14,14,14,14,14,14,14,14,14,
    #                     14,14,14,14,14,14,14,14,14,
    #                     14,14,4), "14-28-4")
    nn.Train([-1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1], [3])

    # print("Input: {0}\nOutput: {1}".format(lvInput, lvOutput))