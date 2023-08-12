import numpy as np

class Affine:
    def __init__(self, input_size, output_size):
        print("Hello, Affine class")
        self.w = np.ones((input_size, output_size))
        self.b = 1.2

    def forward(self, x):
        print("forward")
        return np.dot(x, self.w) + self.b

    def backward(self):
        print("backward")


class BinaryCrossEntropy:
    def __init__(self):
        print("Hello, BinaryCrossEntropy")

    def forward(self, x):
        print("forward")
        return x

    def backward(self):
        print("backward")
