import csv, os, numpy as np

DATA_ROOT_DIR = os.path.join(os.path.dirname(__file__), 'b')
FLOAT_TYPE = 'float32'
PRECISION = "%.16f"

# Cross entropy


class NeuralNetwork:

    synLyrCount = 0
    shape = None
    name = ""
    sigma = None
    nOutput = 0

    # Matrices
    V = [] # array of row vectors, No. layers
    Z = [] # array of row vectors, No. layers - 1

    W = [] # array of weights matrix, No. layers - 1
    B = [] # array of row vectors, No. layers - 1

    dW = [] # array of d_weights matrix, No. layers - 1
    dB = [] # array of d_biases matrix, No. layers - 1

    def __init__(self, shape, name):

        self.synLyrCount = len(shape) - 1
        self.shape = shape
        self.name = name
        self.sigma = np.vectorize(ReLu)
        self.nOutput = int(self.shape[-1])

        self.importWeightsFromCsv()
        self.importBiasesFromCsv()

    def ForwardPropagation(self, X):
        self.V = []
        self.V.append(X)
        for index in range(self.synLyrCount):
            Zi = self.V[index].dot(self.W[index]) + self.B[index]
            self.Z.append(Zi)

            if index < self.synLyrCount - 1:
                self.V.append(self.sigma(Zi))
            else:
                self.V.append(SoftMax(Zi))
            # print self.V[index + 1]

        return self.V[-1]

    def BackPropagation(self, O, Y):

        # dC / dW = (O-Y)/N * dO / dW

        dO = SoftMax(O, True).flatten() # do / dz
        xentC = (O-Y).flatten()
        dO_dW = []
        dO_dB = []

        for k in range(self.nOutput):
            do_dZ = [None] * self.synLyrCount # array of column vectors, No. layers - 1
            doi_dW = [None] * self.synLyrCount # array of weights matrices for oi, No. layers - 1
            doi_dB = [None] * self.synLyrCount # array of biases matrices for oi, No. layers - 1

            # calculate doi / dw
            for index in reversed(range(self.synLyrCount)):
                if index == self.synLyrCount - 1:
                    op = [0] * self.nOutput
                    op[k] = 1 # only keey kth doi / dz
                    do_dZ[index] = np.array([op]).T # a column vector of doi / dz
                else:
                    dZi = self.W[index + 1].dot(do_dZ[index + 1]) # a column vector of doi / dz
                    dSigma = self.sigma(self.Z[index], True).T # a column vector of sigma derivative
                    do_dZ[index] = dSigma * dZi # Hadamard product
                
                doi_dW[index] = self.V[index].T.dot(do_dZ[index].T) # calculate doi / dw from v and doi / dz
                doi_dB[index] = do_dZ[index].T

            dO_dW.append(doi_dW * np.array([xentC[k]])) # save weights matrices for oi with cross entropy label constant
            dO_dB.append(doi_dB * np.array([xentC[k]]))

        # reduce and calculate final dC / dW
        self.dW = np.sum(dO_dW, axis=0)
        self.dB = np.sum(dO_dB, axis=0)

        self.ExportBiasesToCsv()
        self.ExportWeightsToCsv()

    def Train(self, vInput, vLabel):
        X = np.array([vInput])
        Y = self.LabelMapper(vLabel)
        O = self.ForwardPropagation(X)
        self.BackPropagation(O, Y)

    def Run(self, vInput):
        X = np.array([vInput])
        self.ForwardPropagation(X)

    def TrainEpoch(self, trainingSet, trainingRate = 0.2):
        pass

        # delta = []
        # lnCases = input.shape[0]

        # self.Run(input)

        # for index in reversed(range(self.layerCount)):
        #     if index == self.layerCount - 1:
        #         output_delta = self._layerOutput[index] - target.T
        #         error = np.sum(output_delta**2)
        #         delta.append(output_delta * self.sgm(self._layerInput[index], True)
        #     else:
        #         delta_pullback = self.weights[index + 1].T.dot(delta[-1])


    def LabelMapper(self, label):

        if len(label) == 1:
            l = [0] * self.nOutput
            try:
                l[int(label[0])] = 1
            finally:
                l[-1] = 1
            return np.array([l])
        return []

    def importWeightsFromCsv(self):
        with open(os.path.join(DATA_ROOT_DIR, "w-{0}.csv".format(self.name)), 'rb') as csvFile:
            lineReader = csv.reader(csvFile)
            lyr = 0
            ln = 0
            offset = 0
            weight = []
            offset += int(self.shape[lyr])
            for row in lineReader:
                if ln < offset:
                    weight.append(row[1:])
                else:
                    self.W.append(np.array(weight).astype(FLOAT_TYPE))
                    lyr += 1
                    offset += int(self.shape[lyr])
                    weight = [row[1:]]
                ln += 1
            self.W.append(np.array(weight).astype(FLOAT_TYPE))

    def importBiasesFromCsv(self):
        with open(os.path.join(DATA_ROOT_DIR, "b-{0}.csv".format(self.name)), 'rb') as csvFile:
            lineReader = csv.reader(csvFile)
            # if len(self.shape) > 0:
            #     biases.append([0]*self.shape[0])
            for row in lineReader:
                self.B.append(np.array([row[1:]]).astype(FLOAT_TYPE))

    
    def ExportBiasesToCsv(self):

        with open("db-{0}.csv".format(self.name), 'wb') as csvFile:
            lineWriter = csv.writer(csvFile)
            for block in self.dB:
                for row in block:
                    lineWriter.writerow([PRECISION % x for x in row])

    def ExportWeightsToCsv(self):

        with open("dw-{0}.csv".format(self.name), 'wb') as csvFile:
            lineWriter = csv.writer(csvFile)
            for block in self.dW:
                for row in block:
                    lineWriter.writerow([PRECISION % x for x in row])
                


def Sigmoid(x, Derivative = False):
    if Derivative:
        out = Sigmoid(x)
        return out * (1 - out)
    else:
        return 1 / (1 + np.exp(-x))

def ReLu(x, Derivative = False):
    if Derivative:
        return 1 if x > 0 else 0
    else:
        return np.maximum(0,x)

def SoftMax(x,  Derivative = False):
    if Derivative:
        # cross-entropy derivative i = j
        out = SoftMax(x)
        return out * (1 - out)
    else:
        xshift = x - np.max(x)
        exps = np.exp(xshift)
        return exps / np.sum(exps, axis=1, keepdims=True)

def Output(x):
    return 1 if x > 0 else 0