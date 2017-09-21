import numpy as np
from init import loadBias,loadWeight,relu,softmax

# import matplotlib.pyplot as plt

y_train_raw = np.genfromtxt('../data/y_train.csv', delimiter=",").astype(np.int64)
y_train = np.eye(4)[y_train_raw].astype(np.int64)
x_train = np.genfromtxt('../data/x_train.csv', delimiter=",").astype(np.int64)
b_100_40_4 = []
w_100_40_4 = []
loadBias(b_100_40_4, '../data/b-100-40-4.csv')
loadWeight(w_100_40_4,'../data/w-100-40-4.csv',[14,100,40,4])
print w_100_40_4[0]
print w_100_40_4[0].shape
print w_100_40_4[1]
print w_100_40_4[1].shape
print w_100_40_4[2]
print w_100_40_4[2].shape

# print w_100_40_4[1]
# print w_100_40_4[2]




