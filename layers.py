import numpy as np


class Affine:
    def __init__(self, input_size, output_size):
        print("Hello, Affine class")
        self.w = np.ones((input_size, output_size))
        self.b = 1.2

    def forward(self, x):
        print("Affine forward")
        return np.dot(x, self.w) + self.b

    def backward(self):
        print("Affine backward")


class Sigmoid:
    def __init__(self):
        print("Hello, Sigmoid class")
        self.out = None

    def forward(self, x):
        print("Sigmoid forward")
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self):
        print("Sigmoid backward")


class Softmax:
    def __init__(self):
        print("Hello, Softmax")

    def forward(self, x):
        print("Softmax forward")
        exp_sum = np.sum(np.exp(x), axis=1)
        for i in range(x.shape[0]):
            x[i, :] = np.exp(x[i, :]) / exp_sum[i]

        return x

    def backward(self):
        print("Softmax backward")


class BinaryCrossEntropy:
    def __init__(self):
        print("Hello, BinaryCrossEntropy")

    def forward(self, x, label):
        print("BinaryCrossEntropy forward")
        loss_sum = 0
        for i in range(x.shape[0]):
            if label[i] == 1:
                prob = x[i, :][0]
            else:
                prob = x[i, :][1]
            loss_sum += label[i] * np.log(prob) + (1 - label[i]) * np.log(1 - prob)

        return (-1) * loss_sum / x.shape[0]

    def backward(self):
        print("BinaryCrossEntropy backward")
