import argparse
import csv
import json
from layers import Affine, BinaryCrossEntropy, Sigmoid, Softmax
from network import MultilayerPerceptron


def main(test_data_path: str, param_path: str):
    with open(test_data_path) as f:
        reader = csv.reader(f, delimiter=",")
        input_data_list = [row for row in reader]

    with open(param_path) as f:
        param_data = json.load(f)

    params = param_data["params"]
    layers = []
    loss_layer = None
    for param in params:
        print(param["layer"])
        layer = param["layer"]
        if layer == "Affine":
            layers.append(Affine(param["input_size"], param["output_size"], param["lr_rate"]))
        elif layer == "Sigmoid":
            layers.append(Sigmoid())
        elif layer == "Softmax":
            layers.append(Softmax(param["input_size"], param["output_size"]))
        elif layer == "BinaryCrossEntropy":
            loss_layer = BinaryCrossEntropy()

    batch_size = param_data["batch_size"]
    net = MultilayerPerceptron(layers, loss_layer, batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_path")
    parser.add_argument("--param_path")
    args = parser.parse_args()

    main(args.test_data_path, args.param_path)
