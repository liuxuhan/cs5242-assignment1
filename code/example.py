# Python imports
import numpy as np  # Matrix and vector computation package
import matplotlib.pyplot as plt  # Plotting library


# Define the non-linear functions used
def logistic(z):
    return 1 / (1 + np.exp(-z))


def logistic_deriv(y):  # Derivative of logistic function
    return np.multiply(y, (1 - y))


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)


class Layer(object):
    def get_params_iter(self):
        return []

    def get_params_grad(self, X, output_grad):
        return []

    def get_output(self, X):
        pass

    def get_input_grad(self, Y, output_grad=None, T=None):
        pass


class LinearLayer(Layer):
    """The linear layer performs a linear transformation to its input."""

    def __init__(self, n_in, n_out):
        """Initialize hidden layer parameters.
        n_in is the number of input variables.
        n_out is the number of output variables."""
        self.W = np.random.randn(n_in, n_out) * 0.1
        self.b = np.zeros(n_out)

    def get_params_iter(self):
        """Return an iterator over the parameters."""
        return itertools.chain(np.nditer(self.W, op_flags=['readwrite']),
                               np.nditer(self.b, op_flags=['readwrite']))

    def get_output(self, X):
        """Perform the forward step linear transformation."""
        return X.dot(self.W) + self.b

    def get_params_grad(self, X, output_grad):
        """Return a list of gradients over the parameters."""
        JW = X.T.dot(output_grad)
        Jb = np.sum(output_grad, axis=0)
        return [g for g in itertools.chain(np.nditer(JW), np.nditer(Jb))]

    def get_input_grad(self, Y, output_grad):
        """Return the gradient at the inputs of this layer."""
        return output_grad.dot(self.W.T)