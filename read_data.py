from keras.datasets import fashion_mnist

def get_fashion_mnist():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    return (x_train, y_train), (x_test, y_test)
