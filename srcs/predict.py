import argparse
import csv
import json
import random

import numpy as np
from evaluate import calculate_accuracy
from layers import Affine, BinaryCrossEntropy, Sigmoid, Softmax
from network import MultilayerPerceptron
from normalize import get_norm_feature_list
from preprocess import extract_input_data


def main(test_data_path: str, param_path: str):
    with open(test_data_path) as f:
        reader = csv.reader(f, delimiter=",")
        input_data_list = [row for row in reader]

    test_feature_list, test_label_list = extract_input_data(input_data_list)
    normed_test_feature_list = get_norm_feature_list(test_feature_list)

    with open(param_path) as f:
        param_data = json.load(f)

    params = param_data["params"]
    layers = []
    loss_layer = None
    for param in params:
        layer = param["layer"]
        if layer == "Affine":
            affine_layer = Affine(
                param["input_size"], param["output_size"], param["lr_rate"]
            )
            affine_layer.w = np.array(param["weights"])
            affine_layer.b = np.array(param["bias"])
            layers.append(affine_layer)
        elif layer == "Sigmoid":
            layers.append(Sigmoid())
        elif layer == "Softmax":
            layers.append(Softmax(param["input_size"], param["output_size"]))
        elif layer == "BinaryCrossEntropy":
            loss_layer = BinaryCrossEntropy()

    batch_size = param_data["batch_size"]
    net = MultilayerPerceptron(layers, loss_layer, batch_size)
    test_loss = net.calculate_loss(
        np.array(normed_test_feature_list[:batch_size]),
        np.array(test_label_list[:batch_size]),
        layers,
        False,
    )
    accuracy = calculate_accuracy(
        net, layers, batch_size, normed_test_feature_list, test_label_list
    )
    print("Loss: ", round(test_loss, 4))
    print("Accuracy: {0}%".format(round(accuracy * 100, 1)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data_path")
    parser.add_argument("--param_path")
    args = parser.parse_args()

    random.seed(10)
    np.random.seed(10)
    main(args.test_data_path, args.param_path)
