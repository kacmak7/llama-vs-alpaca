import numpy as np

# activation function
def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

# 3 layer neural network
class NeuralNetwork:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.weights1   = np.random.rand(self.x.shape[1],4)
        self.weights2   = np.random.rand(4,1)
        self.output     = np.zeros(self.y.shape)

    def fit(self, epochs):
        self.epochs = epochs

        def feedforward():
            self.layer1 = sigmoid(np.dot(self.x, self.weights1))
            self.output = sigmoid(np.dot(self.layer1, self.weights2))
            print(self.output)

        def backpropagation():
            d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
            d_weights1 = np.dot(self.x.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

            self.weights1 += d_weights1
            self.weights2 += d_weights2

        for i in range(epochs):
            feedforward()
            backpropagation()

