

import numpy as np

from lib.utils import load_graph

bike1_train_data=np.load("./data/ST-SSL_Dataset/NYCBike1/train.npz")
bike1_val_data=np.load("./data/ST-SSL_Dataset/NYCBike1/val.npz")
bike1_test_data=np.load("./data/ST-SSL_Dataset/NYCBike1/test.npz")
train_x=bike1_train_data['x']
train_y=bike1_train_data['y']
train_x_offsets=bike1_train_data['x_offsets']
train_y_offsets=bike1_train_data['y_offsets']
g=load_graph("./data/ST-SSL_Dataset/NYCBike1/adj_mx.npz")

print(train_x.shape)
print(train_y)

