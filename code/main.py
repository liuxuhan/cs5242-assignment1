import numpy as np
from init import loadBias

# import matplotlib.pyplot as plt

y_train_raw = np.genfromtxt('../data/y_train.csv', delimiter=",").astype(np.int64)
y_train = np.eye(4)[y_train_raw].astype(np.int64)
x_train = np.genfromtxt('../data/x_train.csv', delimiter=",").astype(np.int64)
b_100_40_4 = []
loadBias(b_100_40_4, '../data/b-100-40-4.csv')

print b_100_40_4[0]
print b_100_40_4[1]
print b_100_40_4[2]




