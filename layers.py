import numpy as np


class Affine:
    def __init__(self, input_size, output_size):
        print("Hello, Affine class")
        self.x = None
        self.w = np.random.randn(input_size, output_size) / np.sqrt(input_size)
        rng = np.random.default_rng()
        self.b = rng.random()

    def forward(self, x):
        print("Affine forward")
        self.x = x
        return np.dot(x, self.w) + self.b

    def backward(self, dx):
        print("Affine backward")
        dw = np.dot(dx, self.w.T)
        # TODO: Understand why shape is different
        print(dw)
        print(dw.shape)
        print(self.w)
        print(self.w.shape)
        input()

        # db = np.sum(dx)


class Sigmoid:
    def __init__(self):
        print("Hello, Sigmoid class")
        self.out = None

    def forward(self, x):
        print("Sigmoid forward")
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, dx):
        print("Sigmoid backward")
        res = dx * (1 - self.out) * self.out
        return res


class Softmax:
    def __init__(self, input_size, output_size):
        print("Hello, Softmax")
        self.x = None
        self.dx = np.zeros((input_size, output_size))

    def forward(self, x):
        print("Softmax forward")
        exp_sum = np.sum(np.exp(x), axis=1)
        for i in range(x.shape[0]):
            x[i, :] = np.exp(x[i, :]) / exp_sum[i]

        self.x = x
        return x

    def backward(self, label):
        print("Softmax backward")
        for i in range(self.x.shape[0]):
            if label[i] == 1:
                self.dx[i] = np.array([self.x[i, 0]-1, self.x[i, 1]-0])
            else:
                self.dx[i] = np.array([self.x[i, 0]-0, self.x[i, 1]-1])

        return self.dx


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

    # def backward(self):
    #     print("BinaryCrossEntropy backward")
    #     print(self.x)
    #     print(self.label)
    #     input()
    #     y - t
