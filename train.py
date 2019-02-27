import numpy as np
import read_data
import neural_network

(x_train, y_train), (x_test, y_test) = read_data.get_fashion_mnist()

nn = neural_network.NeuralNetwork(x_train, y_train)
nn.fit(1000)