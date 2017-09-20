import numpy as np 
# import matplotlib.pyplot as plt

y_train_raw = np.genfromtxt('../y_train.csv',delimiter=",").astype(np.int64)
y_train = np.eye(4)[y_train_raw].astype(np.int64)
x_train = np.genfromtxt('../x_train.csv',delimiter=",").astype(np.int64)

def softmax(z):
	return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

def ReLU(z):
	return np.maximum(z, 0, z)


