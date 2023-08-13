

class MultilayerPerceptron:
    def __init__(self, layers, loss_layer):
        print("Hello, MultilayerPerceptron")
        self.layers = layers
        self.loss_layer = loss_layer

    def predict(self, x, layers):
        input_arr = x
        for layer in layers:
            res = layer.forward(input_arr)
            print("res: ", res)
            input_arr = res

        return res

    def calculate_loss(self, x, label, layers):
        res = self.predict(x, layers)
        return self.loss_layer.forward(res, label)
